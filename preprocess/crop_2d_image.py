"""
crop_2d_image.py

功能：
1. 从 /root/autodl-tmp/posed_images 读取 .sens 提取后的 RGB-D posed frames；
2. 从 /root/autodl-tmp/mask3d_inst_seg_pcds 读取 Mask3D proposal 点云；
3. 从 /root/autodl-tmp/scans 读取 axisAlignment；
4. 为每个 proposal 离线构建 crop pool，每个 proposal 最多保存 20 张 crop；
5. 保存 view_meta，用于后续在线阶段根据 target / anchor / relation 动态选 5 张图拼 canvas。

输入目录：
    /root/autodl-tmp/posed_images
    /root/autodl-tmp/mask3d_inst_seg_pcds
    /root/autodl-tmp/scans

输出目录：
    /root/autodl-tmp/crop_images
    /root/autodl-tmp/view_meta_scanrefer
    /root/autodl-tmp/crop_pool_meta_scanrefer
"""

import json
import os
import cv2
import numpy as np
import mmengine

from collections import defaultdict
from functools import partial
import psutil
import time


# =========================
# Paths
# =========================

POSED_IMAGE_ROOT = "/root/autodl-tmp/posed_images"
MASK3D_ROOT = "/root/autodl-tmp/mask3d_inst_seg_pcds"
SCANS_ROOT = "/root/autodl-tmp/scans"

OUTPUT_ROOT = "/root/autodl-tmp"

CROP_IMAGE_ROOT = os.path.join(OUTPUT_ROOT, "crop_images")
VIEW_META_ROOT = os.path.join(OUTPUT_ROOT, "view_meta_scanrefer")
CROP_POOL_META_ROOT = os.path.join(OUTPUT_ROOT, "crop_pool_meta_scanrefer")


# =========================
# Config
# =========================

# 离线每个 proposal 最多保存多少张 crop
MAX_CROP_POOL_SIZE = 20

# 在线阶段每个 query-proposal pair 再动态选 5 张，这里只用于记录设计，不在本脚本中拼 canvas
ONLINE_CANVAS_K = 5

# strong crop：目标较清楚，优先进入 pool
MIN_STRONG_CROP_SIZE = 20
MIN_STRONG_VISIBLE_POINTS = 20

# weak crop：目标较小，但作为 fallback 也保存
MIN_WEAK_CROP_SIZE = 8
MIN_WEAK_VISIBLE_POINTS = 8

# crop 周围 padding，越大上下文越多
PAD_RATIO = 0.25

# depth 可见性阈值
VISIBILITY_THRESHOLD = 0.25
CUT_NUM_PIXEL_BOUNDARY = 0

# 是否跳过已处理 scene
SKIP_EXISTING_SCENE = True

# 是否清理旧 crop，避免混入旧结果
CLEAR_OLD_SCENE_CROPS = True

# 并发数：I/O 重，建议 1 或 2
NPROC = 1


# =========================
# Scene / file helpers
# =========================

def get_scene_layout(scene_dir):
    """
    目前主要支持 .sens 提取后的 flat posed_images：
        scene/
          00000.jpg
          00000.png
          00000.txt
          intrinsic.txt

    也兼容 frames25k：
        scene/
          color/
          depth/
          pose/
    """
    has_flat = (
        os.path.isfile(os.path.join(scene_dir, "intrinsic.txt"))
        or os.path.isfile(os.path.join(scene_dir, "intrinsics.txt"))
    )

    has_frames25k = (
        os.path.isdir(os.path.join(scene_dir, "color"))
        and os.path.isdir(os.path.join(scene_dir, "depth"))
        and os.path.isdir(os.path.join(scene_dir, "pose"))
    )

    if has_frames25k:
        return "frames25k"
    if has_flat:
        return "flat"
    return None


def get_image_list(scene_dir, scene_layout):
    """
    返回同时有 RGB / depth / pose 的 frame id。
    """
    if scene_layout == "frames25k":
        color_dir = os.path.join(scene_dir, "color")
        depth_dir = os.path.join(scene_dir, "depth")
        pose_dir = os.path.join(scene_dir, "pose")

        color_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(color_dir)
            if f.endswith((".jpg", ".png")) and os.path.splitext(f)[0].isdigit()
        }
        depth_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(depth_dir)
            if f.endswith(".png") and os.path.splitext(f)[0].isdigit()
        }
        pose_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(pose_dir)
            if f.endswith(".txt") and os.path.splitext(f)[0].isdigit()
        }

        return sorted(color_stems & depth_stems & pose_stems, key=lambda x: int(x))

    fs = os.listdir(scene_dir)

    rgb_stems = {
        os.path.splitext(f)[0]
        for f in fs
        if f.endswith(".jpg") and os.path.splitext(f)[0].isdigit()
    }

    depth_stems = {
        os.path.splitext(f)[0]
        for f in fs
        if f.endswith(".png") and os.path.splitext(f)[0].isdigit()
    }

    pose_stems = {
        os.path.splitext(f)[0]
        for f in fs
        if f.endswith(".txt")
        and os.path.splitext(f)[0].isdigit()
        and f not in ["intrinsic.txt", "intrinsics.txt"]
    }

    return sorted(rgb_stems & depth_stems & pose_stems, key=lambda x: int(x))


def find_intrinsic_path(scene_dir, scene_layout):
    if scene_layout == "frames25k":
        candidates = [
            os.path.join(scene_dir, "intrinsics_color"),
            os.path.join(scene_dir, "intrinsics_color.txt"),
            os.path.join(scene_dir, "intrinsic_color.txt"),
            os.path.join(scene_dir, "intrinsics.txt"),
            os.path.join(scene_dir, "intrinsic.txt"),
            os.path.join(scene_dir, "intrinsic", "intrinsic_color.txt"),
        ]
    else:
        candidates = [
            os.path.join(scene_dir, "intrinsic.txt"),
            os.path.join(scene_dir, "intrinsics.txt"),
            os.path.join(scene_dir, "intrinsic_color.txt"),
            os.path.join(scene_dir, "intrinsics_color.txt"),
        ]

    for p in candidates:
        if os.path.isfile(p):
            return p

    return None


def find_axis_file(scene_id):
    """
    支持以下几种 scans 结构：
    /root/autodl-tmp/scans/scene0000_00/scene0000_00.txt
    /root/autodl-tmp/scans/txt/scene0000_00.txt
    /root/autodl-tmp/scans/txt/scene0000_00/scene0000_00.txt
    """
    candidates = [
        os.path.join(SCANS_ROOT, scene_id, f"{scene_id}.txt"),
        os.path.join(SCANS_ROOT, "txt", f"{scene_id}.txt"),
        os.path.join(SCANS_ROOT, "txt", scene_id, f"{scene_id}.txt"),
    ]

    for p in candidates:
        if os.path.isfile(p):
            return p

    # 兜底：递归搜索，避免目录结构多一层
    for root, _, files in os.walk(SCANS_ROOT):
        target = f"{scene_id}.txt"
        if target in files:
            return os.path.join(root, target)

    return None


def load_axis_align_matrix(scene_id):
    axis_file = find_axis_file(scene_id)
    if axis_file is None:
        raise FileNotFoundError(f"Cannot find axisAlignment txt for {scene_id} under {SCANS_ROOT}")

    with open(axis_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("axisAlignment"):
                vals = line.rstrip().replace("axisAlignment = ", "").split()
                vals = [float(x) for x in vals]
                return np.array(vals, dtype=np.float32).reshape(4, 4)

    raise ValueError(f"axisAlignment not found in {axis_file}")


def find_mask3d_npz(scene_id):
    """
    支持以下几种结构：
    /root/autodl-tmp/mask3d_inst_seg_pcds/scene0000_00.npz
    /root/autodl-tmp/mask3d_inst_seg_pcds/scannet200/scene0000_00.npz
    /root/autodl-tmp/mask3d_inst_seg_pcds/mask3d_inst_seg_pcds/scene0000_00.npz
    """
    candidates = [
        os.path.join(MASK3D_ROOT, f"{scene_id}.npz"),
        os.path.join(MASK3D_ROOT, "scannet200", f"{scene_id}.npz"),
        os.path.join(MASK3D_ROOT, "mask3d_inst_seg_pcds", f"{scene_id}.npz"),
        os.path.join(MASK3D_ROOT, "mask3d", "scannet200", f"{scene_id}.npz"),
    ]

    for p in candidates:
        if os.path.isfile(p):
            return p

    # 兜底递归搜索
    target = f"{scene_id}.npz"
    for root, _, files in os.walk(MASK3D_ROOT):
        if target in files:
            return os.path.join(root, target)

    return None


def get_scene_list():
    """
    优先从 posed_images 目录读取 scene。
    也可以手动改成读取 subset 文件。
    """
    scenes = [
        x for x in os.listdir(POSED_IMAGE_ROOT)
        if x.startswith("scene") and os.path.isdir(os.path.join(POSED_IMAGE_ROOT, x))
    ]
    return sorted(set(scenes))


def is_scene_already_done(scene_id):
    meta_path = os.path.join(VIEW_META_ROOT, f"{scene_id}.json")
    crop_scene_dir = os.path.join(CROP_IMAGE_ROOT, scene_id)

    if not os.path.isfile(meta_path):
        return False

    if not os.path.isdir(crop_scene_dir):
        return False

    for _, _, files in os.walk(crop_scene_dir):
        if any(f.endswith(".jpg") for f in files):
            return True

    return False


def clear_scene_crops(scene_id):
    crop_scene_dir = os.path.join(CROP_IMAGE_ROOT, scene_id)
    if not os.path.isdir(crop_scene_dir):
        return

    for root, _, files in os.walk(crop_scene_dir):
        for f in files:
            if f.endswith(".jpg"):
                try:
                    os.remove(os.path.join(root, f))
                except Exception:
                    pass


# =========================
# Projection
# =========================

class PointCloudToImageMapper(object):

    def __init__(self, visibility_threshold, cut_bound):
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound

    def compute_mapping(self, camera_to_world, coords, depth, image_dim, intrinsic):
        """
        3D points -> 2D pixels.

        Args:
            camera_to_world: 4 x 4 camera pose
            coords: N x 3 world-coordinate points
            depth: H x W depth in meters
            image_dim: (W, H)
            intrinsic: 3 x 3

        Returns:
            mapping: N x 3, [row, col, valid_mask]
        """
        n = coords.shape[0]
        mapping = np.zeros((3, n), dtype=np.int32)

        if n == 0:
            return mapping.T

        coords_h = np.concatenate(
            [coords, np.ones([n, 1], dtype=coords.dtype)],
            axis=1
        ).T

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_h)

        z = p[2]
        front_mask = z > 1e-6
        z_safe = np.where(front_mask, z, 1.0)

        u = (p[0] * intrinsic[0][0]) / z_safe + intrinsic[0][2]
        v = (p[1] * intrinsic[1][1]) / z_safe + intrinsic[1][2]

        pi_u = np.round(u).astype(np.int32)
        pi_v = np.round(v).astype(np.int32)

        w, h = image_dim

        inside_mask = (
            front_mask
            & (pi_u >= self.cut_bound)
            & (pi_v >= self.cut_bound)
            & (pi_u < w - self.cut_bound)
            & (pi_v < h - self.cut_bound)
        )

        if depth is not None and inside_mask.any():
            valid_indices = np.where(inside_mask)[0]
            du = pi_u[valid_indices]
            dv = pi_v[valid_indices]

            depth_cur = depth[dv, du]
            valid_depth = depth_cur > 0

            diff = np.abs(depth_cur - z[valid_indices])
            occlusion_mask = valid_depth & (
                diff <= self.vis_thres * np.maximum(depth_cur, 1e-6)
            )

            new_inside = np.zeros_like(inside_mask)
            new_inside[valid_indices] = occlusion_mask
            inside_mask = new_inside

        mapping[0][inside_mask] = pi_v[inside_mask]
        mapping[1][inside_mask] = pi_u[inside_mask]
        mapping[2][inside_mask] = 1

        return mapping.T


def load_image(scene_dir, image_name, scene_layout):
    if scene_layout == "frames25k":
        img_path_jpg = os.path.join(scene_dir, "color", image_name + ".jpg")
        img_path_png = os.path.join(scene_dir, "color", image_name + ".png")
        img_path = img_path_jpg if os.path.isfile(img_path_jpg) else img_path_png

        depth_path = os.path.join(scene_dir, "depth", image_name + ".png")
        pose_path = os.path.join(scene_dir, "pose", image_name + ".txt")
    else:
        stem = os.path.join(scene_dir, image_name)
        img_path = stem + ".jpg"
        depth_path = stem + ".png"
        pose_path = stem + ".txt"

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {img_path}")

    image_dim = (img.shape[1], img.shape[0])

    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Depth not found or unreadable: {depth_path}")

    depth = depth_raw.astype(np.float32) / 1000.0

    if depth.shape[1] != image_dim[0] or depth.shape[0] != image_dim[1]:
        depth = cv2.resize(depth, image_dim, interpolation=cv2.INTER_NEAREST)

    pose = np.loadtxt(pose_path).astype(np.float32)

    return img, depth, pose, image_dim


# =========================
# Crop helpers
# =========================

def crop_with_padding(img, bbox_xyxy, image_dim, pad_ratio=PAD_RATIO):
    x1, y1, x2, y2 = bbox_xyxy

    crop_w = x2 - x1
    crop_h = y2 - y1

    pad_w = int(crop_w * pad_ratio)
    pad_h = int(crop_h * pad_ratio)

    cx1 = max(x1 - pad_w, 0)
    cy1 = max(y1 - pad_h, 0)
    cx2 = min(x2 + pad_w, image_dim[0])
    cy2 = min(y2 + pad_h, image_dim[1])

    crop_img = img[cy1:cy2, cx1:cx2]

    return crop_img, [int(cx1), int(cy1), int(cx2), int(cy2)]


def get_view_quality(crop_w, crop_h, visible_points):
    """
    strong / weak 只表示当前 proposal 在该帧是否看得清楚，
    不表示是否包含参照物。
    """
    if (
        crop_h >= MIN_STRONG_CROP_SIZE
        and crop_w >= MIN_STRONG_CROP_SIZE
        and visible_points >= MIN_STRONG_VISIBLE_POINTS
    ):
        return "strong"

    if (
        crop_h >= MIN_WEAK_CROP_SIZE
        and crop_w >= MIN_WEAK_CROP_SIZE
        and visible_points >= MIN_WEAK_VISIBLE_POINTS
    ):
        return "weak"

    return "none"


# =========================
# Main process
# =========================

def process_scene(scene_id, point2img_mapper):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    _ = process.cpu_percent(interval=None)
    start_mem = process.memory_info().rss / (1024 * 1024)

    print(f"Processing {scene_id} ...")

    if SKIP_EXISTING_SCENE and is_scene_already_done(scene_id):
        print(f"[Skip existing] {scene_id}")
        return

    scene_dir = os.path.join(POSED_IMAGE_ROOT, scene_id)
    if not os.path.isdir(scene_dir):
        print(f"[Skip] Missing posed image scene dir: {scene_dir}")
        return

    scene_layout = get_scene_layout(scene_dir)
    if scene_layout is None:
        print(f"[Skip] Unknown scene layout: {scene_dir}")
        return

    image_list = get_image_list(scene_dir, scene_layout)
    print(f"Number of images: {len(image_list)}")

    if len(image_list) == 0:
        print(f"[Skip] No valid rgb/depth/pose frames for {scene_id}")
        return

    npz_path = find_mask3d_npz(scene_id)
    if npz_path is None:
        print(f"[Skip] Missing Mask3D npz for {scene_id} under {MASK3D_ROOT}")
        return

    data = np.load(npz_path, allow_pickle=True)

    if "ins_pcds" not in data:
        print(f"[Skip] npz has no ins_pcds: {npz_path}")
        return

    ins_pcds = data["ins_pcds"]

    try:
        axis_align_matrix = load_axis_align_matrix(scene_id)
    except Exception as e:
        print(f"[Skip] Failed to load axisAlignment for {scene_id}: {e}")
        return

    axis_inv_matrix = np.linalg.inv(axis_align_matrix)

    intrinsic_path = find_intrinsic_path(scene_dir, scene_layout)
    if intrinsic_path is None:
        print(f"[Skip] Missing intrinsic file in {scene_dir}")
        return

    intrinsic = np.loadtxt(intrinsic_path).astype(np.float32)
    if intrinsic.shape == (4, 4):
        intrinsic = intrinsic[:3, :3]

    if intrinsic.shape != (3, 3):
        print(f"[Skip] Invalid intrinsic shape {intrinsic.shape}: {intrinsic_path}")
        return

    all_views = defaultdict(list)
    view_meta = defaultdict(list)
    debug_stats = defaultdict(int)

    for image_name in image_list:
        try:
            img, depth, pose, image_dim = load_image(scene_dir, image_name, scene_layout)
        except Exception as e:
            print(f"[Warn] Failed to load frame {scene_id}/{image_name}: {e}")
            continue

        for obj_id, pts in enumerate(ins_pcds):
            debug_stats["total_obj_frame"] += 1

            pts = pts[:, :3]
            if pts.shape[0] == 0:
                debug_stats["empty_points"] += 1
                continue

            # Mask3D proposal 一般在 axis-aligned 坐标系中，
            # 这里恢复到原始 ScanNet 坐标系，用于和 camera pose 对齐。
            pts_h = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)))
            orig_pts = np.dot(pts_h, axis_inv_matrix.T)[:, :3]

            mapping = point2img_mapper.compute_mapping(
                camera_to_world=pose,
                coords=orig_pts,
                depth=depth,
                image_dim=image_dim,
                intrinsic=intrinsic,
            )

            valid_map = mapping[mapping[:, -1] != 0]

            visible_points = int(valid_map.shape[0])
            if visible_points < MIN_WEAK_VISIBLE_POINTS:
                debug_stats["too_few_visible_points"] += 1
                continue

            rows = valid_map[:, 0]
            cols = valid_map[:, 1]

            y1, y2 = int(rows.min()), int(rows.max())
            x1, x2 = int(cols.min()), int(cols.max())

            crop_h = y2 - y1
            crop_w = x2 - x1

            quality = get_view_quality(crop_w, crop_h, visible_points)
            if quality == "none":
                debug_stats["crop_too_small"] += 1
                continue

            bbox_xyxy = [int(x1), int(y1), int(x2), int(y2)]

            crop_img, crop_xyxy = crop_with_padding(
                img=img,
                bbox_xyxy=bbox_xyxy,
                image_dim=image_dim,
                pad_ratio=PAD_RATIO,
            )

            if crop_img is None or crop_img.size == 0:
                debug_stats["empty_crop"] += 1
                continue

            obj_key = str(obj_id)
            area = float(crop_h * crop_w)
            image_area = float(image_dim[0] * image_dim[1])
            area_ratio = float(area / max(image_area, 1.0))

            item = {
                "crop_img": crop_img,
                "image_name": str(image_name),
                "area": area,
                "area_ratio": area_ratio,
                "visible_points": visible_points,
                "bbox_xyxy": bbox_xyxy,
                "crop_xyxy": crop_xyxy,
                "quality": quality,
                "crop_w": int(crop_w),
                "crop_h": int(crop_h),
                "image_width": int(image_dim[0]),
                "image_height": int(image_dim[1]),
            }

            all_views[obj_key].append(item)

            if quality == "strong":
                debug_stats["strong"] += 1
            else:
                debug_stats["weak"] += 1

            view_meta[obj_key].append({
                "scene_id": scene_id,
                "obj_id": int(obj_id),
                "image_name": str(image_name),

                # 在 full-frame 上的 bbox，后续画红框/蓝框直接用这个，不需要模板匹配
                "bbox_xyxy": bbox_xyxy,

                # 实际 crop 范围
                "crop_xyxy": crop_xyxy,

                "area": area,
                "area_ratio": area_ratio,
                "visible_points": visible_points,
                "quality": quality,
                "crop_w": int(crop_w),
                "crop_h": int(crop_h),
                "image_width": int(image_dim[0]),
                "image_height": int(image_dim[1]),
            })

    all_obj_keys = sorted(all_views.keys(), key=lambda x: int(x))

    print(f"Valid proposal objects: {len(all_obj_keys)}")
    print(f"[debug] {scene_id}: {dict(debug_stats)}")

    if CLEAR_OLD_SCENE_CROPS:
        clear_scene_crops(scene_id)

    os.makedirs(CROP_IMAGE_ROOT, exist_ok=True)
    os.makedirs(VIEW_META_ROOT, exist_ok=True)
    os.makedirs(CROP_POOL_META_ROOT, exist_ok=True)

    # 保存 view_meta：后续动态选图、target-anchor 同屏判断、画框都靠它
    meta_path = os.path.join(VIEW_META_ROOT, f"{scene_id}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(view_meta, f, indent=2)
    print(f"Saved view metadata to {meta_path}")

    pool_meta = defaultdict(list)
    pool_count_hist = defaultdict(int)
    num_saved_obj = 0

    crop_scene_root = os.path.join(CROP_IMAGE_ROOT, scene_id)
    os.makedirs(crop_scene_root, exist_ok=True)

    for obj_id in all_obj_keys:
        views = all_views[obj_id]

        # strong 优先，同级按面积和可见点数排序
        views_sorted = sorted(
            views,
            key=lambda x: (
                1 if x["quality"] == "strong" else 0,
                x["area_ratio"],
                x["visible_points"],
            ),
            reverse=True,
        )

        selected_pool = views_sorted[:MAX_CROP_POOL_SIZE]

        if len(selected_pool) == 0:
            continue

        obj_path = os.path.join(crop_scene_root, str(obj_id))
        os.makedirs(obj_path, exist_ok=True)

        # 清理该 object 旧图
        for old_f in os.listdir(obj_path):
            if old_f.endswith(".jpg"):
                try:
                    os.remove(os.path.join(obj_path, old_f))
                except Exception:
                    pass

        for rank, item in enumerate(selected_pool):
            crop_img = item["crop_img"]
            image_name = item["image_name"]
            quality = item["quality"]

            # 保留 rank + quality + 原始 frame_id，便于后续回溯
            out_name = f"img_{rank:03d}_{quality}_{image_name}.jpg"
            out_path = os.path.join(obj_path, out_name)

            cv2.imwrite(out_path, crop_img)

            pool_meta[obj_id].append({
                "rank": int(rank),
                "file_name": out_name,
                "image_name": str(image_name),
                "quality": str(quality),
                "area": float(item["area"]),
                "area_ratio": float(item["area_ratio"]),
                "visible_points": int(item["visible_points"]),
                "bbox_xyxy": item["bbox_xyxy"],
                "crop_xyxy": item["crop_xyxy"],
                "crop_w": int(item["crop_w"]),
                "crop_h": int(item["crop_h"]),
                "image_width": int(item["image_width"]),
                "image_height": int(item["image_height"]),
            })

        num_saved_obj += 1
        pool_count_hist[len(selected_pool)] += 1

    pool_meta_path = os.path.join(CROP_POOL_META_ROOT, f"{scene_id}.json")
    with open(pool_meta_path, "w", encoding="utf-8") as f:
        json.dump(pool_meta, f, indent=2)
    print(f"Saved crop pool metadata to {pool_meta_path}")

    end_time = time.time()
    end_mem = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent(interval=None)

    print(f"[done] {scene_id}")
    print(f"[done] saved proposal dirs: {num_saved_obj}")
    print(f"[done] pool count hist: {dict(sorted(pool_count_hist.items()))}")
    print(
        f"[resource] cpu={cpu_percent:.2f}%, "
        f"mem_delta={end_mem - start_mem:.2f}MB, "
        f"time={end_time - start_time:.2f}s"
    )


# =========================
# Main
# =========================

if __name__ == "__main__":
    print(f"[info] POSED_IMAGE_ROOT = {POSED_IMAGE_ROOT}")
    print(f"[info] MASK3D_ROOT = {MASK3D_ROOT}")
    print(f"[info] SCANS_ROOT = {SCANS_ROOT}")
    print(f"[info] OUTPUT_ROOT = {OUTPUT_ROOT}")
    print(f"[info] CROP_IMAGE_ROOT = {CROP_IMAGE_ROOT}")
    print(f"[info] VIEW_META_ROOT = {VIEW_META_ROOT}")
    print(f"[info] CROP_POOL_META_ROOT = {CROP_POOL_META_ROOT}")
    print(f"[info] MAX_CROP_POOL_SIZE = {MAX_CROP_POOL_SIZE}")
    print(f"[info] ONLINE_CANVAS_K = {ONLINE_CANVAS_K}")

    scene_list = get_scene_list()
    print(f"[info] num scenes found in posed_images = {len(scene_list)}")

    point2img_mapper = PointCloudToImageMapper(
        visibility_threshold=VISIBILITY_THRESHOLD,
        cut_bound=CUT_NUM_PIXEL_BOUNDARY,
    )

    process_func = partial(
        process_scene,
        point2img_mapper=point2img_mapper,
    )

    mmengine.track_parallel_progress(
        func=process_func,
        tasks=scene_list,
        nproc=NPROC,
    )