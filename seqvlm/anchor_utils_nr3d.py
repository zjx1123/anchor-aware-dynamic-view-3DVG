import numpy as np
from typing import List, Dict, Any


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


def build_anchor_candidates_nr3d(
    anchors: List[Dict[str, Any]],
    obj_ids: List[int],
    pred_class_list: List[str],
    ins_locs,
    obj_embeds,
    handler,
    seg_conf_score: float = 0.0,
    max_anchor_per_type: int = 5,
) -> List[Dict[str, Any]]:
    """
    NR3D anchor candidates.

    NR3D 映射规则：
    - local_index: feats_3d.pkl 中数组下标，用于 ins_locs / crop / view_meta / dynamic canvas
    - proposal_id: 给 DynamicViewSelector 使用，必须等于 local_index
    - obj_id: NR3D / ReferIt3D 原始 object id，只用于日志和对齐检查
    """
    anchor_infos = []

    if not anchors:
        return anchor_infos

    for anchor in anchors:
        anchor_cat = anchor.get("category", "")
        if not anchor_cat:
            continue

        try:
            matched_cls, _, anchor_match_score = \
                handler.predict_obj_class(
                    anchor_cat,
                    obj_embeds,
                    return_match_score=True,
                )
        except Exception as e:
            print(
                f"[Anchor Match Error] "
                f"anchor={anchor_cat}, error={e}"
            )
            continue

        # 新增：低质量参照物文本匹配直接丢弃
        if anchor_match_score < seg_conf_score:
            print(
                f"[Anchor Filtered] "
                f"anchor={anchor_cat}, "
                f"matched_class={matched_cls}, "
                f"clip_score={anchor_match_score:.4f}, "
                f"threshold={seg_conf_score:.4f}"
            )
            continue

        candidates = []

        for local_idx, cls_name in enumerate(pred_class_list):
            if cls_name != matched_cls:
                continue

            obj_id = int(obj_ids[local_idx])

            # NR3D feats_3d.pkl 这里没有 segmentation confidence，
            # 用 1.0 作为中性分数。
            score = 1.0
            if score < seg_conf_score:
                continue

            candidates.append({
                "anchor_query_category": anchor_cat,
                "matched_class": matched_cls,

                # 关键：proposal_id 给 dynamic canvas / view_meta / crop_pool 用
                "proposal_id": int(local_idx),

                # NR3D 原始 object id 单独保存
                "obj_id": int(obj_id),

                "local_index": int(local_idx),
                "score": float(score),
                "loc": ins_locs[local_idx],
                "relation_to_target": anchor.get("relation_to_target", ""),
                "attributes": anchor.get("attributes", []),
            })

        candidates = sorted(
            candidates,
            key=lambda x: (-float(x["score"]), int(x["local_index"]))
        )

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
            f"- anchor proposal_id={a['proposal_id']}, "
            f"obj_id={a.get('obj_id', 'NA')}, "
            f"local_index={a.get('local_index', 'NA')}, "
            f"query_category={a['anchor_query_category']}, "
            f"matched_class={a['matched_class']}, "
            f"relation_to_target={a.get('relation_to_target', '')}, "
            f"attributes={attr_text}, "
            f"{loc_text}"
        )

    return "\n".join(lines)


def build_target_candidate_meta_nr3d(
    target_local_indices: List[int],
    target_obj_ids: List[int],
    pred_class_list: List[str],
    ins_locs,
    scores=None,
) -> List[Dict[str, Any]]:
    """
    local_image_id: VLM 看到的 image_id，对应 prop_images 的全局候选序号
    proposal_id: local_index，用于 dynamic canvas / crop_pool / view_meta
    obj_id: NR3D 原始 object id
    """
    metas = []

    if scores is None:
        scores = [1.0] * len(target_local_indices)

    for local_image_id, (local_idx, obj_id) in enumerate(
        zip(target_local_indices, target_obj_ids)
    ):
        metas.append({
            "local_image_id": int(local_image_id),

            # 关键：proposal_id 必须是 local_index
            "proposal_id": int(local_idx),

            # 原始 NR3D object id 单独保存
            "obj_id": int(obj_id),

            "local_index": int(local_idx),
            "label": pred_class_list[local_idx],
            "score": float(scores[local_image_id]),
            "loc": ins_locs[local_idx],
        })

    return metas


def summarize_candidate_batch_for_prompt_nr3d(
    batch_items,
    candidate_metas: List[Dict[str, Any]],
    anchor_infos: List[Dict[str, Any]],
) -> str:
    """
    batch_items: [(global_candidate_idx, base64_image), ...]
    global_candidate_idx 对应 prop_images 中的全局候选 id。
    """
    lines = []
    lines.append("Candidate target objects in the current image batch:")

    meta_by_local = {
        m["local_image_id"]: m for m in candidate_metas
    }

    for image_id, _ in batch_items:
        m = meta_by_local.get(image_id)
        if m is None:
            continue

        lines.append(
            f"- image_id={image_id}, "
            f"proposal_id={m['proposal_id']}, "
            f"obj_id={m.get('obj_id', 'NA')}, "
            f"local_index={m.get('local_index', 'NA')}, "
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
                    a.get("obj_id", "NA"),
                    a["matched_class"],
                    a.get("relation_to_target", ""),
                ))

            dists = sorted(dists, key=lambda x: x[0])[:5]

            dist_text = "; ".join([
                f"to anchor proposal_id={aid}, obj_id={obj_id}({cls}, rel={rel}): {dist:.2f}m"
                for dist, aid, obj_id, cls, rel in dists
            ])

            lines.append(f"  nearest anchors: {dist_text}")

    return "\n".join(lines)