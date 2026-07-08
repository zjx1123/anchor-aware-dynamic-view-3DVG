'''
用 PyTorch3D 从 ScanNet 对齐 mesh 渲染 SPAZER 风格的全局 5 视角（
top / down / up / left / right）。
'''
import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from plyfile import PlyData

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader,
    TexturesVertex,
    BlendParams,
    AmbientLights,
)


VIEW_ORDER = ["top", "down", "up", "left", "right"]


def read_scene_list(path):
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]


def load_colored_mesh(ply_path, device):
    ply = PlyData.read(ply_path)

    v = ply["vertex"]
    verts = np.stack(
        [
            np.asarray(v["x"]),
            np.asarray(v["y"]),
            np.asarray(v["z"]),
        ],
        axis=1,
    ).astype(np.float32)

    names = v.data.dtype.names
    if "red" in names and "green" in names and "blue" in names:
        colors = np.stack(
            [
                np.asarray(v["red"]),
                np.asarray(v["green"]),
                np.asarray(v["blue"]),
            ],
            axis=1,
        ).astype(np.float32) / 255.0
    else:
        colors = np.ones((verts.shape[0], 3), dtype=np.float32) * 0.7

    if "face" not in ply:
        raise RuntimeError(f"No face element in mesh: {ply_path}")

    faces_raw = ply["face"].data["vertex_indices"]
    faces = np.vstack(faces_raw).astype(np.int64)

    verts_t = torch.from_numpy(verts).float().to(device)
    faces_t = torch.from_numpy(faces).long().to(device)
    colors_t = torch.from_numpy(colors).float().to(device)

    textures = TexturesVertex(verts_features=[colors_t])
    mesh = Meshes(verts=[verts_t], faces=[faces_t], textures=textures)

    return mesh, verts


def compute_scene_camera_params(
    verts,
    view_name,
    fov=60.0,
    alpha_deg=45.0,
    dist_scale=1.0,
):
    """
    SPAZER-style global view camera.

    top:
        BEV view, camera directly above scene center.

    down/up/left/right:
        Four oblique top-down views distributed around the scene.
        The oblique angle alpha is 45 degrees by default.

    The base camera distance follows SPAZER:
        d = 0.5 * max(lx, ly) / tan(theta / 2)
    """

    xyz_min = verts.min(axis=0)
    xyz_max = verts.max(axis=0)

    center = (xyz_min + xyz_max) / 2.0
    extent = xyz_max - xyz_min

    lx, ly, lz = extent.tolist()
    cx, cy, cz = center.tolist()

    theta = np.deg2rad(fov)
    base_d = 0.5 * max(lx, ly) / np.tan(theta / 2.0)

    # dist_scale 默认 1.0，表示尽量贴近 SPAZER 公式
    d = float(base_d * dist_scale)

    alpha = np.deg2rad(alpha_deg)
    horizontal_r = d * np.cos(alpha)
    z_offset = d * np.sin(alpha)

    if view_name == "top":
        eye = [cx, cy, cz + d]
        at = [cx, cy, cz]
        up = [0.0, 1.0, 0.0]

    else:
        # 为了兼容已有命名：
        # down = front-oblique, 从 y 负方向斜俯视
        # up = back-oblique, 从 y 正方向斜俯视
        # left = left-oblique, 从 x 负方向斜俯视
        # right = right-oblique, 从 x 正方向斜俯视
        if view_name == "down":
            phi = -np.pi / 2.0
            semantic_view = "front_oblique"
        elif view_name == "up":
            phi = np.pi / 2.0
            semantic_view = "back_oblique"
        elif view_name == "left":
            phi = np.pi
            semantic_view = "left_oblique"
        elif view_name == "right":
            phi = 0.0
            semantic_view = "right_oblique"
        else:
            raise ValueError(f"Unknown view_name: {view_name}")

        eye = [
            cx + horizontal_r * np.cos(phi),
            cy + horizontal_r * np.sin(phi),
            cz + z_offset,
        ]
        at = [cx, cy, cz]
        up = [0.0, 0.0, 1.0]

    return {
        "eye": eye,
        "at": at,
        "up": up,
        "center": center.tolist(),
        "extent": extent.tolist(),
        "base_distance_spazer": float(base_d),
        "distance_used": float(d),
        "fov": float(fov),
        "alpha_deg": float(alpha_deg),
        "view_style": "bev" if view_name == "top" else semantic_view,
    }


def make_renderer(device, image_size=1024, faces_per_pixel=1):
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=faces_per_pixel,
        cull_backfaces=False,
    )

    blend_params = BlendParams(
        background_color=(1.0, 1.0, 1.0)
    )

    # ambient light 保留 vertex color，避免阴影过重
    lights = AmbientLights(
        device=device,
        ambient_color=((1.0, 1.0, 1.0),),
    )

    def build(cameras):
        return MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings,
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights,
                blend_params=blend_params,
            ),
        )

    return build


def render_one_view(mesh, verts_np, view_name, device, image_size=1024, fov=60.0, alpha_deg=45.0, dist_scale=1.0):
    cam_param = compute_scene_camera_params(
        verts=verts_np,
        view_name=view_name,
        fov=fov,
        alpha_deg=alpha_deg,
        dist_scale=dist_scale,
    )

    eye = torch.tensor([cam_param["eye"]], dtype=torch.float32, device=device)
    at = torch.tensor([cam_param["at"]], dtype=torch.float32, device=device)
    up = torch.tensor([cam_param["up"]], dtype=torch.float32, device=device)

    R, T = look_at_view_transform(
        eye=eye,
        at=at,
        up=up,
    )

    cameras = FoVPerspectiveCameras(
        device=device,
        R=R,
        T=T,
        fov=fov,
    )

    renderer_builder = make_renderer(
        device=device,
        image_size=image_size,
        faces_per_pixel=1,
    )
    renderer = renderer_builder(cameras)

    with torch.no_grad():
        image = renderer(mesh)[0, ..., :3]

    image_np = (image.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image_np)

    meta = {
        "view_name": view_name,
        "image_width": image_size,
        "image_height": image_size,
        "camera": {
            "eye": cam_param["eye"],
            "at": cam_param["at"],
            "up": cam_param["up"],
            "fov": fov,
            "alpha_deg": alpha_deg,
            "R": R.detach().cpu().numpy()[0].tolist(),
            "T": T.detach().cpu().numpy()[0].tolist(),
        },
        "scene": {
            "center": cam_param["center"],
            "extent": cam_param["extent"],
            "base_distance_spazer": cam_param["base_distance_spazer"],
            "distance_used": cam_param["distance_used"],
            "view_style": cam_param["view_style"],
        },
    }

    return pil_img, meta


def add_label(img, text):
    img = img.convert("RGB")
    canvas = img.copy()

    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([10, 10, 180, 52], fill=(255, 255, 255))
    draw.text((20, 22), text, fill=(0, 0, 0))

    return canvas


def build_scene_rendered_views(
    scene_id,
    mesh_root,
    out_root,
    device,
    image_size=1024,
    fov=60.0,
    alpha_deg=45.0,
    dist_scale=1.0,
    overwrite=False,
):
    mesh_path = os.path.join(mesh_root, f"{scene_id}.ply")
    out_dir = os.path.join(out_root, scene_id)
    os.makedirs(out_dir, exist_ok=True)

    stitched_path = os.path.join(out_dir, "stitched_horizontal.png")
    meta_path = os.path.join(out_dir, "render_meta.json")

    if os.path.exists(stitched_path) and os.path.exists(meta_path) and not overwrite:
        print(f"[skip existing] {scene_id}")
        return True

    if not os.path.exists(mesh_path):
        print(f"[skip] missing mesh: {mesh_path}")
        return False

    print(f"[render] {scene_id}")
    mesh, verts_np = load_colored_mesh(mesh_path, device=device)

    view_imgs = []
    view_metas = []

    for slot, view_name in enumerate(VIEW_ORDER):
        img, meta = render_one_view(
            mesh=mesh,
            verts_np=verts_np,
            view_name=view_name,
            device=device,
            image_size=image_size,
            fov=fov,
            alpha_deg=alpha_deg,
            dist_scale=dist_scale,
        )

        img = add_label(img, f"view_{slot}: {view_name}")

        img_name = f"_mesh_{view_name}.png"
        img_path = os.path.join(out_dir, img_name)
        img.save(img_path)

        meta["slot"] = slot
        meta["image_path"] = img_name
        view_metas.append(meta)
        view_imgs.append(img)

        # 释放一点显存
        torch.cuda.empty_cache() if device.type == "cuda" else None

    stitched = Image.new(
        "RGB",
        (image_size * len(view_imgs), image_size),
        (255, 255, 255),
    )

    for i, img in enumerate(view_imgs):
        stitched.paste(img, (i * image_size, 0))

    stitched.save(stitched_path)

    render_meta = {
        "scene_id": scene_id,
        "views": view_metas,
        "view_order": VIEW_ORDER,
        "stitched_image": "stitched_horizontal.png",
        "note": "Rendered from axis-aligned ScanNet mesh using PyTorch3D mesh renderer.",
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(render_meta, f, indent=2)

    print(f"[done] {scene_id}: {out_dir}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_list", required=True)
    parser.add_argument(
        "--mesh_root",
        default=os.path.join(DATA_ROOT, "global_aligned_mesh_clean_2"),
    )
    parser.add_argument(
        "--out_root",
        default=os.path.join(DATA_ROOT, "global_rendered_views_scanrefer"),
    )
    parser.add_argument("--scene_id", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--dist_scale", type=float, default=1.35)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--alpha_deg", type=float, default=45.0)
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
            if build_scene_rendered_views(
                scene_id=scene_id,
                mesh_root=args.mesh_root,
                out_root=args.out_root,
                device=device,
                image_size=args.image_size,
                fov=args.fov,
                dist_scale=args.dist_scale,
                overwrite=args.overwrite,
                alpha_deg=args.alpha_deg,
            ):
                ok += 1
        except RuntimeError as e:
            print(f"[error] {scene_id}: {e}")
            if "out of memory" in str(e).lower():
                print("[hint] Try --image_size 768 or 512, or reduce fov/dist_scale.")
            torch.cuda.empty_cache() if device.type == "cuda" else None

    print(f"[summary] ok = {ok} / {len(scenes)}")


if __name__ == "__main__":
    main()