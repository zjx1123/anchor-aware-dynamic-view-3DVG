
import os
import numpy as np
from typing import List, Dict, Any, Tuple


def bbox_center_size_to_text(loc) -> str:
    """
    loc: [cx, cy, cz, sx, sy, sz]
    """
    cx, cy, cz, sx, sy, sz = [float(x) for x in loc]
    return (
        f"center=({cx:.2f}, {cy:.2f}, {cz:.2f}), "
        f"size=({sx:.2f}, {sy:.2f}, {sz:.2f})"
    )


def l2_distance(loc_a, loc_b) -> float:
    return float(np.linalg.norm(np.array(loc_a[:3]) - np.array(loc_b[:3])))


def build_anchor_candidates(
    anchors: List[Dict[str, Any]],
    ins_labels: List[str],
    ins_locs,
    ins_scores,
    handler,
    image_path: str,
    scene_id: str,
    seg_conf_score: float = 0.2,
    max_anchor_per_type: int = 5,
) -> List[Dict[str, Any]]:
    """
    Return:
    [
      {
        "anchor_query_category": "table",
        "matched_class": "table",
        "proposal_id": 12,
        "score": 0.91,
        "loc": [...],
        "canvas": ".../canvas.jpg"
      }
    ]
    """
    anchor_infos = []

    if not anchors:
        return anchor_infos

    for anchor in anchors:
        anchor_cat = anchor.get("category", "")
        if not anchor_cat:
            continue

        try:
            matched_cls = handler.predict_obj_class(anchor_cat, ins_labels)
        except Exception:
            matched_cls = anchor_cat

        candidates = []
        for i, label in enumerate(ins_labels):
            if ins_scores[i] <= seg_conf_score:
                continue
            if label != matched_cls:
                continue

            # 注意：dynamic view 不再依赖 anchor 的固定 canvas。
            # anchor 的 visual evidence 后续由 view_meta_scanrefer + posed_images_rgb_pose 动态生成。
            candidates.append({
                "anchor_query_category": anchor_cat,
                "matched_class": matched_cls,
                "proposal_id": i,
                "score": float(ins_scores[i]),
                "loc": ins_locs[i],
                "relation_to_target": anchor.get("relation_to_target", ""),
                "attributes": anchor.get("attributes", []),
            })

        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        anchor_infos.extend(candidates[:max_anchor_per_type])

    return anchor_infos


def summarize_anchors_for_prompt(anchor_infos: List[Dict[str, Any]]) -> str:
    if not anchor_infos:
        return "No explicit anchor/reference objects were parsed from the query."

    lines = []
    lines.append("Parsed reference/anchor objects in this scene:")
    for a in anchor_infos:
        loc_text = bbox_center_size_to_text(a["loc"])
        attr_text = ", ".join(a.get("attributes", [])) or "none"
        lines.append(
            f"- anchor proposal id={a['proposal_id']}, "
            f"query_category={a['anchor_query_category']}, "
            f"matched_class={a['matched_class']}, "
            f"relation_to_target={a.get('relation_to_target', '')}, "
            f"attributes={attr_text}, "
            f"{loc_text}"
        )

    return "\n".join(lines)


def build_target_candidate_meta(
    target_indices: List[int],
    ins_labels: List[str],
    ins_locs,
    ins_scores,
) -> List[Dict[str, Any]]:
    metas = []
    for local_id, proposal_id in enumerate(target_indices):
        metas.append({
            "local_image_id": local_id,
            "proposal_id": proposal_id,
            "label": ins_labels[proposal_id],
            "score": float(ins_scores[proposal_id]),
            "loc": ins_locs[proposal_id],
        })
    return metas


def summarize_candidate_batch_for_prompt(
    batch_items,
    candidate_metas: List[Dict[str, Any]],
    anchor_infos: List[Dict[str, Any]],
) -> str:
    """
    batch_items: [(global_candidate_idx, base64_image), ...]
    global_candidate_idx 对应 prop_images 的 index。
    """
    lines = []
    lines.append("Candidate target objects in the current image batch:")

    meta_by_local = {
        m["local_image_id"]: m
        for m in candidate_metas
    }

    for image_id, _ in batch_items:
        m = meta_by_local.get(image_id)
        if m is None:
            continue

        lines.append(
            f"- image_id={image_id}, "
            f"proposal_id={m['proposal_id']}, "
            f"label={m['label']}, "
            f"score={m['score']:.3f}, "
            f"{bbox_center_size_to_text(m['loc'])}"
        )

        if anchor_infos:
            dists = []
            for a in anchor_infos:
                dists.append((
                    l2_distance(m["loc"], a["loc"]),
                    a["proposal_id"],
                    a["matched_class"],
                    a.get("relation_to_target", ""),
                ))
            dists = sorted(dists, key=lambda x: x[0])[:5]
            dist_text = "; ".join(
                [
                    f"to anchor_id={aid}({cls}, rel={rel}): {dist:.2f}m"
                    for dist, aid, cls, rel in dists
                ]
            )
            lines.append(f"  nearest anchors: {dist_text}")

    return "\n".join(lines)