#!/usr/bin/env python3
"""Run reproducible component ablations for ScanRefer or NR3D.

Each configuration gets independent dynamic-canvas/global-view directories,
its exact command, a full console log, parsed metrics, and a final CSV/Markdown
table.  Run this file from anywhere inside the repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQVLM_DIR = PROJECT_ROOT / "seqvlm"


# The configurations intentionally respect module dependencies.
# AQ: anchor-aware query parsing/reasoning
# TC: target-centric adaptive views
# AR: target-anchor relational views (requires AQ + TC)
# FG: gated final-round global evidence (requires AQ)
CONFIGS = {
    "baseline": {
        "label": "Baseline (fixed canvas)",
        "AQ": False, "TC": False, "AR": False, "FG": False,
    },
    "anchor": {
        "label": "+ Anchor-aware reasoning",
        "AQ": True, "TC": False, "AR": False, "FG": False,
    },
    "target": {
        "label": "+ Target-centric views",
        "AQ": False, "TC": True, "AR": False, "FG": False,
    },
    "anchor_target": {
        "label": "+ Anchor reasoning + target-centric views",
        "AQ": True, "TC": True, "AR": False, "FG": False,
    },
    "local_full": {
        "label": "+ Relational views (full local model)",
        "AQ": True, "TC": True, "AR": True, "FG": False,
    },
    "full": {
        "label": "+ Gated final-global evidence (full model)",
        "AQ": True, "TC": True, "AR": True, "FG": True,
    },
}


DATASET_ARGS = {
    "nr3d": {
        "entry": "evaluate_nr3d.py",
        "data_path": "../data/nr3d_250.json",
        "image_path": "../data/nr3d_preprocessed",
        "crop_image_root": "../data/crop_images_nr3d",
        "crop_pool_meta_root": "../data/crop_pool_meta_nr3d",
        "view_meta_root": "../data/view_meta_nr3d",
        "posed_image_root": "../data/posed_images_rgb_pose",
        "global_rendered_root": "../data/global_rendered_views_nr3d",
        "seg_conf_score": "0.85",
        "metric_names": ["Overall@25", "Overall@50", "Easy", "Hard", "VD", "VID"],
    },
    "scanrefer": {
        "entry": "evaluate.py",
        "data_path": "../data/scanrefer_250.json",
        "image_path": "../data/scanrefer_preprocessed",
        "crop_image_root": "../data/crop_images",
        "crop_pool_meta_root": "../data/crop_pool_meta_scanrefer",
        "view_meta_root": "../data/view_meta_scanrefer",
        "posed_image_root": "../data/posed_images_rgb_pose",
        "global_rendered_root": "../data/global_rendered_views_scanrefer_spazer",
        "seg_conf_score": "0.2",
        "metric_names": [
            "Overall@25", "Overall@50", "Unique@25", "Unique@50",
            "Multiple@25", "Multiple@50",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AnchorVRA component ablations and summarize metrics."
    )
    parser.add_argument("--dataset", choices=DATASET_ARGS, default="scanrefer")
    parser.add_argument(
        "--configs",
        default=",".join(CONFIGS),
        help=f"Comma-separated subset of: {','.join(CONFIGS)}",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vlm_model", default="qwen")
    parser.add_argument("--max_vlm_props", type=int, default=40)
    parser.add_argument("--max_batch_size", type=int, default=4)
    parser.add_argument("--canvas_k", type=int, default=5)
    parser.add_argument("--num_appearance_views", type=int, default=3)
    parser.add_argument("--num_relation_views", type=int, default=2)
    parser.add_argument("--max_final_global_anchors", type=int, default=2)
    parser.add_argument(
        "--output_root",
        default=None,
        help="Default: <repo>/ablation_runs/<dataset>_<timestamp>",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def selected_configs(raw: str) -> list[str]:
    names = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in names if x not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown configs: {unknown}; valid: {list(CONFIGS)}")
    if not names:
        raise SystemExit("No ablation configuration selected.")
    return names


def build_command(
    args: argparse.Namespace,
    name: str,
    run_dir: Path,
) -> list[str]:
    ds = DATASET_ARGS[args.dataset]
    cfg = CONFIGS[name]
    dynamic_root = run_dir / "dynamic_canvas"
    final_global_root = run_dir / "final_global_view"

    cmd = [
        sys.executable, "-u", ds["entry"],
        "--data_path", ds["data_path"],
        "--exp_name", f"ablation_{args.dataset}_{name}",
        "--image_path", ds["image_path"],
        "--vlm_model", args.vlm_model,
        "--max_vlm_props", str(args.max_vlm_props),
        "--max_batch_size", str(args.max_batch_size),
        "--seed", str(args.seed),
        "--seg_conf_score", ds["seg_conf_score"],
    ]

    if cfg["AQ"]:
        cmd.append("--use_anchor_aware")

    if cfg["TC"]:
        # AR=False creates an appearance-only canvas. The selector fills the
        # remaining K slots with additional target appearance crops.
        relation_count = args.num_relation_views if cfg["AR"] else 0
        appearance_count = args.num_appearance_views if cfg["AR"] else args.canvas_k
        cmd.extend([
            "--use_dynamic_canvas",
            "--crop_image_root", ds["crop_image_root"],
            "--crop_pool_meta_root", ds["crop_pool_meta_root"],
            "--view_meta_root", ds["view_meta_root"],
            "--posed_image_root", ds["posed_image_root"],
            "--dynamic_canvas_root", str(dynamic_root),
            "--canvas_k", str(args.canvas_k),
            "--num_appearance_views", str(appearance_count),
            "--num_relation_views", str(relation_count),
        ])

    if cfg["FG"]:
        cmd.extend([
            "--use_final_global_view",
            "--global_rendered_root", ds["global_rendered_root"],
            "--final_global_view_root", str(final_global_root),
            "--max_final_global_anchors", str(args.max_final_global_anchors),
        ])

    if args.max_samples is not None:
        cmd.extend(["--max_samples", str(args.max_samples)])
    return cmd


def run_and_tee(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=SEQVLM_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return process.wait()


def parse_metrics(log_text: str, metric_names: list[str]) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {}
    for metric in metric_names:
        matches = re.findall(
            rf"(?m)^{re.escape(metric)}:\s*([0-9]+(?:\.[0-9]+)?)\s*$",
            log_text,
        )
        metrics[metric] = float(matches[-1]) if matches else ""

    for metric in ("Except Ratio", "VLM Usage Ratio"):
        matches = re.findall(
            rf"(?m)^{re.escape(metric)}:\s*([^\r\n]+?)\s*$",
            log_text,
        )
        metrics[metric] = matches[-1].strip() if matches else ""
    return metrics


def write_summary(
    output_root: Path,
    dataset: str,
    rows: list[dict[str, object]],
) -> None:
    metric_names = DATASET_ARGS[dataset]["metric_names"]
    fields = ["config", "description", "AQ", "TC", "AR", "FG", "status"]
    fields += metric_names + ["Except Ratio", "VLM Usage Ratio", "log"]

    csv_path = output_root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_root / "summary.md"
    md_fields = ["config", "AQ", "TC", "AR", "FG"] + metric_names
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {dataset.upper()} component ablation\n\n")
        f.write("AQ: anchor-aware reasoning; TC: target-centric adaptive views; ")
        f.write("AR: target-anchor relational views; FG: gated final-global evidence.\n\n")
        f.write("| " + " | ".join(md_fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(md_fields)) + " |\n")
        for row in rows:
            values = []
            for field in md_fields:
                value = row.get(field, "")
                if field in {"AQ", "TC", "AR", "FG"}:
                    value = "✓" if value else ""
                values.append(str(value))
            f.write("| " + " | ".join(values) + " |\n")


def main() -> int:
    args = parse_args()
    names = selected_configs(args.configs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else PROJECT_ROOT / "ablation_runs" / f"{args.dataset}_{timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": args.dataset,
        "created_at": timestamp,
        "arguments": vars(args),
        "configs": {name: CONFIGS[name] for name in names},
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"

    rows: list[dict[str, object]] = []
    print(f"Ablation output: {output_root}")

    for index, name in enumerate(names, start=1):
        cfg = CONFIGS[name]
        run_dir = output_root / f"{index:02d}_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(args, name, run_dir)
        command_text = subprocess.list2cmdline(cmd)
        (run_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")

        print(f"\n[{index}/{len(names)}] {name}: {cfg['label']}")
        print(command_text)

        row: dict[str, object] = {
            "config": name,
            "description": cfg["label"],
            "AQ": cfg["AQ"], "TC": cfg["TC"], "AR": cfg["AR"], "FG": cfg["FG"],
            "status": "dry-run" if args.dry_run else "running",
            "log": str(run_dir / "run.log"),
        }

        if args.dry_run:
            rows.append(row)
            continue

        return_code = run_and_tee(cmd, run_dir / "run.log", env)
        row["status"] = "ok" if return_code == 0 else f"failed({return_code})"
        metrics = parse_metrics(
            (run_dir / "run.log").read_text(encoding="utf-8", errors="replace"),
            DATASET_ARGS[args.dataset]["metric_names"],
        )
        row.update(metrics)
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        rows.append(row)
        write_summary(output_root, args.dataset, rows)

        if return_code != 0 and not args.continue_on_error:
            print(f"Stopped after {name} failed. See {run_dir / 'run.log'}")
            return return_code

    write_summary(output_root, args.dataset, rows)
    print(f"\nDone. Summary: {output_root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
