'''
把 Mask3D 实例 proposal 的 3D 点云 投影到每个渲染视角，
计算 2D bbox、可见性、面积等，供后续候选筛选使用。
'''
import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from pytorch3d.renderer import FoVPerspectiveCameras

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def read_scene_list(path):
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def project_points_to_view(points_np, view_meta, device):
    """
    Project axis-aligned Mask3D proposal points to a rendered PyTorch3D view.
    points_np: [N, 3]
    view_meta: one item from render_meta["views"]
    """
    cam = view_meta["camera"]

    R = torch.tensor(cam["R"], dtype=torch.float32, device=device).unsqueeze(0)
    T = torch.tensor(cam["T"], dtype=torch.float32, device=device).unsqueeze(0)
    fov = float(cam["fov"])

    image_width = int(view_meta["image_width"])
    image_height = int(view_meta["image_height"])

    cameras = FoVPerspectiveCameras(
        device=device,
        R=R,
        T=T,
        fov=fov,
    )

    points = torch.tensor(points_np, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        screen = cameras.transform_points_screen(
            points,
            image_size=torch.tensor([[image_height, image_width]], device=device),
            with_xyflip=True,
        )[0]

        view_points = cameras.get_world_to_view_transform().transform_points(points)[0]

    x = screen[:, 0].detach().cpu().numpy()
    y = screen[:, 1].detach().cpu().numpy()
    z_cam = view_points[:, 2].detach().cpu().numpy()

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z_cam)
        & (z_cam > 0)
        & (x >= 0)
        & (x < image_width)
        & (y >= 0)
        & (y < image_height)
    )

    return x, y, valid, image_width, image_height


def bbox_from_valid_pixels(x, y, valid):
    xv = x[valid]
    yv = y[valid]

    x1 = int(np.floor(xv.min()))
    y1 = int(np.floor(yv.min()))
    x2 = int(np.ceil(xv.max()))
    y2 = int(np.ceil(yv.max()))

    return [x1, y1, x2, y2]


def build_scene_meta(
    scene_id,
    mask3d_root,
    rendered_root,
    device,
    min_visible_points=20,
    min_bbox_size=5,
    overwrite=False,
):
    scene_dir = os.path.join(rendered_root, scene_id)

    render_meta_path = os.path.join(scene_dir, "render_meta.json")
    out_path = os.path.join(scene_dir, "rendered_view_meta.json")
    mask_path = os.path.join(mask3d_root, f"{scene_id}.npz")

    if os.path.exists(out_path) and not overwrite:
        print(f"[skip existing] {scene_id}")
        return True

    if not os.path.exists(render_meta_path):
        print(f"[skip] missing render_meta: {render_meta_path}")
        return False

    if not os.path.exists(mask_path):
        print(f"[skip] missing mask3d npz: {mask_path}")
        return False

    render_meta = load_json(render_meta_path)
    data = np.load(mask_path, allow_pickle=True)

    if "ins_pcds" not in data:
        raise KeyError(
            f"{mask_path} does not contain key 'ins_pcds'. Available keys: {data.files}"
        )

    result = {}

    for obj_id, pts in enumerate(data["ins_pcds"]):
        pts = np.asarray(pts[:, :3], dtype=np.float32)

        xyz_min = pts.min(axis=0)
        xyz_max = pts.max(axis=0)

        center = ((xyz_min + xyz_max) / 2.0).tolist()
        size = (xyz_max - xyz_min).tolist()

        obj_result = {
            "proposal_id": int(obj_id),
            "center_3d": center,
            "size_3d": size,
            "views": {},
        }

        best_view = None
        best_area = -1.0

        for view_meta in render_meta["views"]:
            view_name = view_meta["view_name"]

            x, y, valid, image_width, image_height = project_points_to_view(
                points_np=pts,
                view_meta=view_meta,
                device=device,
            )

            visible_points = int(valid.sum())

            if visible_points < min_visible_points:
                obj_result["views"][view_name] = {
                    "visible": False,
                    "visible_points": visible_points,
                    "slot": int(view_meta["slot"]),
                }
                continue

            bbox = bbox_from_valid_pixels(x, y, valid)
            x1, y1, x2, y2 = bbox

            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)

            area = float(bw * bh)
            area_ratio = area / float(image_width * image_height)

            visible = True
            if bw < min_bbox_size or bh < min_bbox_size:
                visible = False

            view_item = {
                "visible": visible,
                "visible_points": visible_points,
                "bbox_xyxy": bbox,
                "bbox_center_xy": [
                    int((x1 + x2) / 2),
                    int((y1 + y2) / 2),
                ],
                "area": area,
                "area_ratio": area_ratio,
                "image_width": image_width,
                "image_height": image_height,
                "slot": int(view_meta["slot"]),
                "view_style": view_meta.get("scene", {}).get("view_style", view_name),
            }

            obj_result["views"][view_name] = view_item

            if visible and area > best_area:
                best_area = area
                best_view = view_name

        obj_result["best_rendered_view"] = best_view
        obj_result["best_rendered_area"] = best_area if best_area >= 0 else 0.0

        result[str(obj_id)] = obj_result

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[done] {scene_id}: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scene_list", required=True)
    parser.add_argument(
        "--mask3d_root",
        default=os.path.join(DATA_ROOT, "mask3d_inst_seg_pcds"),
    )
    parser.add_argument(
        "--rendered_root",
        default=os.path.join(DATA_ROOT, "global_rendered_views_scanrefer_spazer"),
    )
    parser.add_argument("--scene_id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--min_visible_points", type=int, default=20)
    parser.add_argument("--min_bbox_size", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda not available, fallback to cpu")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.scene_id:
        scenes = [args.scene_id]
    else:
        scenes = read_scene_list(args.scene_list)

    ok = 0

    for scene_id in tqdm(scenes):
        try:
            if build_scene_meta(
                scene_id=scene_id,
                mask3d_root=args.mask3d_root,
                rendered_root=args.rendered_root,
                device=device,
                min_visible_points=args.min_visible_points,
                min_bbox_size=args.min_bbox_size,
                overwrite=args.overwrite,
            ):
                ok += 1
        except Exception as e:
            print(f"[error] {scene_id}: {e}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"[summary] ok = {ok} / {len(scenes)}")


if __name__ == "__main__":
    main()