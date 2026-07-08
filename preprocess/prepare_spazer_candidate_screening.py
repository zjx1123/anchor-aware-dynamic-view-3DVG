'''
为 SPAZER Stage 1/2 准备 VLM 输入
——视角选择 prompt、候选筛选 prompt，并在选定视角上画 candidate bbox。
'''
import os
import json
import argparse
from PIL import Image, ImageDraw

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


VALID_VIEWS = ["top", "down", "up", "left", "right"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_ids(s):
    ids = []
    for x in s.split(","):
        x = x.strip()
        if x == "":
            continue
        ids.append(str(int(x)))
    return ids


def draw_candidates(
    image_path,
    rendered_view_meta,
    candidate_ids,
    view_name,
    out_path,
    max_candidates=None,
):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    drawn = []
    skipped = []

    draw_items = []

    for pid in candidate_ids:
        if pid not in rendered_view_meta:
            skipped.append({
                "proposal_id": pid,
                "reason": "proposal_id_not_found",
            })
            continue

        item = rendered_view_meta[pid]
        view_info = item.get("views", {}).get(view_name, {})

        if not view_info.get("visible", False):
            skipped.append({
                "proposal_id": pid,
                "reason": f"not_visible_in_{view_name}",
            })
            continue

        area = float(view_info.get("area", 0.0))
        draw_items.append((pid, area, view_info))

    # 面积大的先画，便于检查；但最终 JSON 仍保留原 candidate 顺序
    draw_items = sorted(draw_items, key=lambda x: x[1], reverse=True)

    if max_candidates is not None and len(draw_items) > max_candidates:
        skipped_extra = draw_items[max_candidates:]
        draw_items = draw_items[:max_candidates]
        for pid, area, view_info in skipped_extra:
            skipped.append({
                "proposal_id": pid,
                "reason": "exceed_max_candidates_for_debug_image",
                "area": area,
            })

    for pid, area, view_info in draw_items:
        x1, y1, x2, y2 = view_info["bbox_xyxy"]
        cx, cy = view_info["bbox_center_xy"]

        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

        label = str(pid)
        label_w = max(36, len(label) * 12 + 12)
        label_h = 26

        lx1 = max(0, cx - 10)
        ly1 = max(0, cy - 10)
        lx2 = min(img.width - 1, lx1 + label_w)
        ly2 = min(img.height - 1, ly1 + label_h)

        draw.rectangle([lx1, ly1, lx2, ly2], fill=(255, 255, 255))
        draw.text((lx1 + 5, ly1 + 5), label, fill=(255, 0, 0))

        drawn.append({
            "proposal_id": pid,
            "bbox_xyxy": view_info["bbox_xyxy"],
            "bbox_center_xy": view_info["bbox_center_xy"],
            "area": area,
            "visible_points": view_info.get("visible_points", None),
            "slot": view_info.get("slot", None),
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)

    return drawn, skipped


def build_view_selection_prompt(query, render_meta):
    return {
        "task": "SPAZER-style 3D Holistic View Selection",
        "instruction": (
            "You are given a natural language query and a stitched rendered global view "
            "containing five sub-views. Select the single sub-view that is most useful "
            "for observing the target object and its spatial relations. "
            "Do not select the final object. Return only valid JSON."
        ),
        "query": query,
        "view_metadata": [
            {
                "slot": v.get("slot"),
                "view_name": v.get("view_name"),
                "view_style": v.get("scene", {}).get("view_style", v.get("view_name")),
                "image_path": v.get("image_path"),
            }
            for v in render_meta.get("views", [])
        ],
        "return_format": {
            "selected_slot": 1,
            "selected_view_name": "down",
            "reasoning": "..."
        }
    }


def build_candidate_screening_prompt(query, selected_view, candidate_ids, topk):
    return {
        "task": "SPAZER-style Candidate Object Screening",
        "instruction": (
            "You are given a rendered global view with candidate proposal IDs annotated. "
            "Your task is to select the Top-k most likely candidate object IDs according "
            "to the query. This is only coarse candidate screening, not the final decision. "
            "Return only valid JSON."
        ),
        "query": query,
        "selected_view": selected_view,
        "candidate_ids": [int(x) for x in candidate_ids],
        "topk": topk,
        "return_format": {
            "topk_candidate_ids": [4, 1, 5, 3],
            "reasoning": "..."
        }
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--selected_view",
        required=True,
        choices=VALID_VIEWS,
        help="For now, manually specify selected view. Later this will come from VLM view selection.",
    )
    parser.add_argument(
        "--candidate_ids",
        required=True,
        help="Comma-separated candidate proposal IDs, e.g. 1,4,7,9",
    )
    parser.add_argument(
        "--rendered_root",
        default=os.path.join(DATA_ROOT, "global_rendered_views_scanrefer_spazer"),
    )
    parser.add_argument(
        "--out_root",
        default=os.path.join(DATA_ROOT, "spazer_stage12_debug"),
    )
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument(
        "--max_candidates_for_debug_image",
        type=int,
        default=None,
        help="If too many candidates clutter the image, set e.g. 80. Default draws all visible candidates.",
    )

    args = parser.parse_args()

    scene_dir = os.path.join(args.rendered_root, args.scene_id)

    render_meta_path = os.path.join(scene_dir, "render_meta.json")
    rendered_view_meta_path = os.path.join(scene_dir, "rendered_view_meta.json")
    selected_img_path = os.path.join(scene_dir, f"_mesh_{args.selected_view}.png")
    stitched_img_path = os.path.join(scene_dir, "stitched_horizontal.png")

    if not os.path.exists(render_meta_path):
        raise FileNotFoundError(f"missing render_meta: {render_meta_path}")

    if not os.path.exists(rendered_view_meta_path):
        raise FileNotFoundError(f"missing rendered_view_meta: {rendered_view_meta_path}")

    if not os.path.exists(selected_img_path):
        raise FileNotFoundError(f"missing selected image: {selected_img_path}")

    if not os.path.exists(stitched_img_path):
        raise FileNotFoundError(f"missing stitched image: {stitched_img_path}")

    render_meta = load_json(render_meta_path)
    rendered_view_meta = load_json(rendered_view_meta_path)
    candidate_ids = parse_ids(args.candidate_ids)

    out_dir = os.path.join(args.out_root, args.scene_id)
    os.makedirs(out_dir, exist_ok=True)

    annotated_path = os.path.join(
        out_dir,
        f"annotated_{args.selected_view}_candidates.png"
    )

    drawn, skipped = draw_candidates(
        image_path=selected_img_path,
        rendered_view_meta=rendered_view_meta,
        candidate_ids=candidate_ids,
        view_name=args.selected_view,
        out_path=annotated_path,
        max_candidates=args.max_candidates_for_debug_image,
    )

    view_selection_prompt = build_view_selection_prompt(
        query=args.query,
        render_meta=render_meta,
    )

    screening_prompt = build_candidate_screening_prompt(
        query=args.query,
        selected_view=args.selected_view,
        candidate_ids=candidate_ids,
        topk=args.topk,
    )

    case_info = {
        "scene_id": args.scene_id,
        "query": args.query,
        "selected_view": args.selected_view,
        "topk": args.topk,
        "stitched_image_path": stitched_img_path,
        "selected_view_image_path": selected_img_path,
        "annotated_candidate_view_path": annotated_path,
        "candidate_ids_input": [int(x) for x in candidate_ids],
        "candidate_ids_drawn": [int(x["proposal_id"]) for x in drawn],
        "candidate_ids_skipped": skipped,
        "num_input_candidates": len(candidate_ids),
        "num_drawn_candidates": len(drawn),
        "num_skipped_candidates": len(skipped),
        "view_selection_prompt_json": view_selection_prompt,
        "candidate_screening_prompt_json": screening_prompt,
    }

    out_json = os.path.join(out_dir, f"stage12_input_{args.selected_view}.json")
    save_json(case_info, out_json)

    # 单独保存两个 prompt，方便你后面直接喂给 VLM
    save_json(
        view_selection_prompt,
        os.path.join(out_dir, "view_selection_prompt.json"),
    )
    save_json(
        screening_prompt,
        os.path.join(out_dir, f"candidate_screening_prompt_{args.selected_view}.json"),
    )

    print("[done]")
    print(f"annotated image: {annotated_path}")
    print(f"case json: {out_json}")
    print(f"input candidates: {len(candidate_ids)}")
    print(f"drawn candidates: {len(drawn)}")
    print(f"skipped candidates: {len(skipped)}")

    if skipped:
        print("\nfirst skipped candidates:")
        for item in skipped[:20]:
            print(item)

    print("\nfirst drawn candidates:")
    for item in drawn[:20]:
        print(item)


if __name__ == "__main__":
    main()