import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQVLM_DIR = PROJECT_ROOT / "seqvlm"

# 兼容项目当前的非包式 import
sys.path.insert(0, str(SEQVLM_DIR))

from feat_handler_nr3d import VisualFeatHandler


def get_cache_path(cache_dir: Path, caption: str) -> Path:
    key = hashlib.md5(caption.encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.json"


def encode_anchor_categories(handler, categories, batch_size=128):
    """批量计算参照物文本的归一化 CLIP embedding。"""
    all_feats = []

    with torch.no_grad():
        for start in range(0, len(categories), batch_size):
            batch_categories = categories[start:start + batch_size]

            tokens = handler.tokenizer(
                [
                    f"a {category} in a scene"
                    for category in batch_categories
                ],
                padding=True,
                return_tensors="pt",
            )

            tokens = {
                key: value.cuda()
                for key, value in tokens.items()
            }

            feats = handler.clip.get_text_features(**tokens)
            feats = feats / feats.norm(
                p=2,
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-6)

            all_feats.append(feats)

    return torch.cat(all_feats, dim=0)


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "nr3d_250.json"),
    )
    parser.add_argument(
        "--feats_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "feats_3d.pkl"),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(
            PROJECT_ROOT / "data" / "cache" / "query_parse_nr3d"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(
            PROJECT_ROOT / "outputs" / "nr3d_anchor_score_stats"
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    data_path = Path(args.data_path)
    feats_path = Path(args.feats_path)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    if args.max_samples is not None:
        eval_data = eval_data[:args.max_samples]

    with open(feats_path, "rb") as f:
        scene_feats = pickle.load(f)

    # ---------------------------------------------------------
    # 第一步：只从现有 query parser cache 中读取参照物
    # 不会调用 API，也不会修改 cache
    # ---------------------------------------------------------
    anchor_occurrences = []
    missing_cache_cases = []

    for case_idx, task in enumerate(eval_data):
        values = list(task.values())

        if len(values) < 7:
            print(f"[Skip] Unexpected task format: case={case_idx}")
            continue

        (
            scene_id,
            obj_id,
            caption,
            prog_str,
            is_easy,
            is_dep,
            obj_name,
        ) = values[:7]

        cache_path = get_cache_path(cache_dir, caption)

        if not cache_path.exists():
            missing_cache_cases.append({
                "case_idx": case_idx,
                "scene_id": scene_id,
                "caption": caption,
                "cache_path": str(cache_path),
            })
            continue

        with open(cache_path, "r", encoding="utf-8") as f:
            parsed_query = json.load(f)

        anchors = parsed_query.get("anchors", [])

        for anchor_idx, anchor in enumerate(anchors):
            anchor_category = anchor.get("category", "").strip()

            if not anchor_category:
                continue

            anchor_occurrences.append({
                "case_idx": case_idx,
                "anchor_idx": anchor_idx,
                "scene_id": scene_id,
                "gt_obj_id": obj_id,
                "target_name": obj_name,
                "caption": caption,
                "anchor_category": anchor_category,
                "anchor_attributes": json.dumps(
                    anchor.get("attributes", []),
                    ensure_ascii=False,
                ),
                "relation_to_target": anchor.get(
                    "relation_to_target",
                    "",
                ),
            })

    print(f"Total cases: {len(eval_data)}")
    print(f"Cases missing parse cache: {len(missing_cache_cases)}")
    print(f"Parsed anchor occurrences: {len(anchor_occurrences)}")

    if missing_cache_cases:
        save_csv(
            out_dir / "missing_parse_cache.csv",
            missing_cache_cases,
            [
                "case_idx",
                "scene_id",
                "caption",
                "cache_path",
            ],
        )

    if not anchor_occurrences:
        raise RuntimeError(
            "No anchors found. Check query_parse_nr3d cache."
        )

    # ---------------------------------------------------------
    # 第二步：加载当前项目使用的 CLIP
    # feat_handler_nr3d.py 的模型路径依赖当前工作目录
    # ---------------------------------------------------------
    os.chdir(SEQVLM_DIR)
    handler = VisualFeatHandler.get_instance()

    unique_anchor_categories = sorted({
        row["anchor_category"]
        for row in anchor_occurrences
    })

    print(
        f"Unique anchor categories: "
        f"{len(unique_anchor_categories)}"
    )

    anchor_text_feats = encode_anchor_categories(
        handler,
        unique_anchor_categories,
    )

    anchor_to_index = {
        category: idx
        for idx, category in enumerate(unique_anchor_categories)
    }

    # 缓存每个场景中实际出现的预测类别
    scene_class_cache = {}

    result_rows = []

    with torch.no_grad():
        for row_idx, row in enumerate(anchor_occurrences):
            scene_id = row["scene_id"]

            if scene_id not in scene_feats:
                print(f"[Skip] Missing scene feats: {scene_id}")
                continue

            if scene_id not in scene_class_cache:
                obj_embeds = scene_feats[scene_id]["obj_embeds"]

                if not torch.is_tensor(obj_embeds):
                    obj_embeds = torch.as_tensor(obj_embeds)

                obj_embeds = obj_embeds.cuda()

                # 完全复现当前 predict_obj_class 中：
                # proposal embedding -> ScanNet200 类别
                class_logits = torch.matmul(
                    handler.label_lang_infos,
                    obj_embeds.t(),
                )

                proposal_class_indices = class_logits.argmax(dim=0)

                # 当前场景所有 proposal 实际预测到的类别
                scene_class_indices = torch.unique(
                    proposal_class_indices
                )

                scene_class_cache[scene_id] = scene_class_indices

            scene_class_indices = scene_class_cache[scene_id]

            anchor_category = row["anchor_category"]
            anchor_feat_idx = anchor_to_index[anchor_category]

            anchor_feat = anchor_text_feats[
                anchor_feat_idx:anchor_feat_idx + 1
            ]

            scene_label_feats = handler.label_lang_infos[
                scene_class_indices
            ]

            # 参照物文本与场景类别的 cosine similarity
            similarities = torch.matmul(
                anchor_feat,
                scene_label_feats.t(),
            ).squeeze(0)

            top_k = min(2, similarities.numel())

            top_scores, top_positions = torch.topk(
                similarities,
                k=top_k,
            )

            best_position = int(top_positions[0].item())
            best_class_idx = int(
                scene_class_indices[best_position].item()
            )

            best_score = float(top_scores[0].item())

            if top_k >= 2:
                second_score = float(top_scores[1].item())
                score_margin = best_score - second_score
            else:
                second_score = float("nan")
                score_margin = float("nan")

            output_row = dict(row)

            output_row.update({
                "matched_class": (
                    handler.class_name_list[best_class_idx]
                ),
                "anchor_clip_score": best_score,
                "second_clip_score": second_score,
                "score_margin": score_margin,
                "num_scene_classes": int(
                    scene_class_indices.numel()
                ),
            })

            result_rows.append(output_row)

            if (row_idx + 1) % 500 == 0:
                print(
                    f"Processed anchors: "
                    f"{row_idx + 1}/"
                    f"{len(anchor_occurrences)}"
                )

    # 按分数从低到高排列，方便直接查看噪声
    result_rows.sort(
        key=lambda x: float(x["anchor_clip_score"])
    )

    fieldnames = [
        "case_idx",
        "anchor_idx",
        "scene_id",
        "gt_obj_id",
        "target_name",
        "anchor_category",
        "matched_class",
        "anchor_clip_score",
        "second_clip_score",
        "score_margin",
        "num_scene_classes",
        "relation_to_target",
        "anchor_attributes",
        "caption",
    ]

    save_csv(
        out_dir / "anchor_clip_scores.csv",
        result_rows,
        fieldnames,
    )

    save_csv(
        out_dir / "lowest_100_anchor_matches.csv",
        result_rows[:100],
        fieldnames,
    )

    scores = np.asarray([
        float(row["anchor_clip_score"])
        for row in result_rows
    ])

    quantiles = {
        "min": float(np.min(scores)),
        "p01": float(np.quantile(scores, 0.01)),
        "p05": float(np.quantile(scores, 0.05)),
        "p10": float(np.quantile(scores, 0.10)),
        "p20": float(np.quantile(scores, 0.20)),
        "p25": float(np.quantile(scores, 0.25)),
        "median": float(np.quantile(scores, 0.50)),
        "p75": float(np.quantile(scores, 0.75)),
        "p90": float(np.quantile(scores, 0.90)),
        "p95": float(np.quantile(scores, 0.95)),
        "p99": float(np.quantile(scores, 0.99)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }

    # ---------------------------------------------------------
    # 不同阈值下会过滤多少参照物、影响多少 case
    # ---------------------------------------------------------
    thresholds = np.arange(0.40, 0.91, 0.05)

    rows_by_case = defaultdict(list)

    for row in result_rows:
        rows_by_case[int(row["case_idx"])].append(
            float(row["anchor_clip_score"])
        )

    threshold_analysis = []

    for threshold in thresholds:
        retained_count = int(np.sum(scores >= threshold))
        filtered_count = int(np.sum(scores < threshold))

        affected_cases = 0
        empty_anchor_cases = 0

        for case_scores in rows_by_case.values():
            retained_in_case = sum(
                score >= threshold
                for score in case_scores
            )

            if retained_in_case < len(case_scores):
                affected_cases += 1

            if retained_in_case == 0:
                empty_anchor_cases += 1

        threshold_analysis.append({
            "threshold": round(float(threshold), 4),
            "retained_anchors": retained_count,
            "filtered_anchors": filtered_count,
            "retained_anchor_ratio": (
                retained_count / len(scores)
            ),
            "affected_cases": affected_cases,
            "empty_anchor_cases": empty_anchor_cases,
            "total_anchor_cases": len(rows_by_case),
            "empty_anchor_case_ratio": (
                empty_anchor_cases / len(rows_by_case)
            ),
        })

    save_csv(
        out_dir / "threshold_analysis.csv",
        threshold_analysis,
        [
            "threshold",
            "retained_anchors",
            "filtered_anchors",
            "retained_anchor_ratio",
            "affected_cases",
            "empty_anchor_cases",
            "total_anchor_cases",
            "empty_anchor_case_ratio",
        ],
    )

    # ---------------------------------------------------------
    # 按参照物类别统计
    # ---------------------------------------------------------
    category_scores = defaultdict(list)

    for row in result_rows:
        category_scores[row["anchor_category"]].append(
            float(row["anchor_clip_score"])
        )

    category_summary = []

    for category, values in category_scores.items():
        values = np.asarray(values)

        category_summary.append({
            "anchor_category": category,
            "count": len(values),
            "min": float(np.min(values)),
            "p10": float(np.quantile(values, 0.10)),
            "median": float(np.quantile(values, 0.50)),
            "mean": float(np.mean(values)),
            "p90": float(np.quantile(values, 0.90)),
            "max": float(np.max(values)),
        })

    category_summary.sort(
        key=lambda x: (
            float(x["mean"]),
            -int(x["count"]),
        )
    )

    save_csv(
        out_dir / "anchor_category_summary.csv",
        category_summary,
        [
            "anchor_category",
            "count",
            "min",
            "p10",
            "median",
            "mean",
            "p90",
            "max",
        ],
    )

    summary = {
        "total_eval_cases": len(eval_data),
        "cases_missing_parse_cache": len(
            missing_cache_cases
        ),
        "total_anchor_occurrences": len(result_rows),
        "unique_anchor_categories": len(
            unique_anchor_categories
        ),
        "anchor_cases": len(rows_by_case),
        "score_quantiles": quantiles,
        "threshold_analysis": threshold_analysis,
    }

    with open(
        out_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # 画直方图
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 6))
        plt.hist(scores, bins=40)
        plt.xlabel("Anchor CLIP match score")
        plt.ylabel("Anchor occurrence count")
        plt.title("NR3D Anchor CLIP Score Distribution")
        plt.tight_layout()
        plt.savefig(
            out_dir / "anchor_clip_score_histogram.png",
            dpi=200,
        )
        plt.close()

    except ImportError:
        print("[Warning] matplotlib not installed; skip plot.")

    print("\nScore quantiles:")
    print(json.dumps(quantiles, indent=2))

    print("\nThreshold analysis:")
    for item in threshold_analysis:
        print(
            f"threshold={item['threshold']:.2f}, "
            f"filtered={item['filtered_anchors']}, "
            f"retained_ratio="
            f"{item['retained_anchor_ratio']:.3f}, "
            f"empty_anchor_cases="
            f"{item['empty_anchor_cases']}/"
            f"{item['total_anchor_cases']}"
        )

    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()