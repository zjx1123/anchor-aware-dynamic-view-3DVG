import argparse
import ast
import re
import shutil
from pathlib import Path


DEFAULT_LOG_DIR = Path("/root/SeqVLM-clean/logs")
DEFAULT_PROJECT_ROOT = Path("/root/SeqVLM-clean/seqvlm")


def split_cases(log_text):
    pattern = re.compile(
        r"^Case:\s*(\d+)\s*\n(.*?)(?=^Case:\s*\d+\s*\n|\Z)",
        re.MULTILINE | re.DOTALL,
    )

    cases = {}
    for m in pattern.finditer(log_text):
        case_id = int(m.group(1))
        block = "Case: " + str(case_id) + "\n" + m.group(2)
        cases[case_id] = block

    return cases


def extract_line(block, key):
    m = re.search(rf"^{re.escape(key)}:\s*(.*)$", block, re.MULTILINE)
    return m.group(1).strip() if m else ""


def extract_iou(block):
    m = re.search(r"IoU:\s*([0-9.]+)", block)
    return float(m.group(1)) if m else None


def extract_target_images(block):
    m = re.search(r"Target Prop Images:\s*\d+\s*(\[.*?\])", block, re.DOTALL)
    if not m:
        return []

    raw = m.group(1)

    stop_markers = [
        "\n[Invoke]",
        "\n[Truncate]",
        "\n[FinalGlobalView]",
        "\nIoU:",
        "\nVision Lang Model Output:",
        "\nCase:",
    ]

    for marker in stop_markers:
        if marker in raw:
            raw = raw.split(marker)[0]

    try:
        paths = ast.literal_eval(raw)
    except Exception:
        paths = re.findall(r"'([^']*canvas\.jpg)'", raw)

    truncate_match = re.search(r"\[Truncate\]\s*Target proposals:\s*(\d+)\s*->\s*(\d+)", block)
    if truncate_match:
        keep_n = int(truncate_match.group(2))
        paths = paths[:keep_n]

    return paths


def extract_num_target_images(block):
    """
    返回实际参与 tournament 的候选数量。
    优先使用日志里的 Target Prop Images: N。
    如果有 [Truncate] Target proposals: 78 -> 40，则返回 40。
    不能用 extract_target_images 的长度，因为日志可能只打印前 20 个路径。
    """
    m = re.search(r"Target Prop Images:\s*(\d+)", block)
    if not m:
        return len(extract_target_images(block))

    n = int(m.group(1))

    truncate_match = re.search(
        r"\[Truncate\]\s*Target proposals:\s*(\d+)\s*->\s*(\d+)",
        block,
    )
    if truncate_match:
        n = int(truncate_match.group(2))

    return n


def extract_winner_ids(block):
    ids = re.findall(r'"image_id"\s*:\s*(\d+)', block)
    return [int(x) for x in ids]


def extract_global_view(block):
    paths = re.findall(r"\[FinalGlobalView\].*?path=([^\s]+)", block)
    return paths[-1] if paths else None


def proposal_id_from_canvas_path(path_str):
    try:
        return Path(path_str).parent.name
    except Exception:
        return "unknown"


def resolve_path(path_str, project_root):
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def safe_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.exists():
        shutil.copy2(src, dst)
        print(f"[OK] {src} -> {dst}")
        return True
    else:
        print(f"[MISSING] {src}")
        return False


def get_final_selected_id(block):
    target_images = extract_target_images(block)
    winner_ids = extract_winner_ids(block)

    if winner_ids:
        return winner_ids[-1]

    if len(target_images) == 1:
        return 0

    return None


def copy_final_selected_canvas(block, tag, out_case_dir, project_root):
    target_images = extract_target_images(block)
    final_id = get_final_selected_id(block)

    if final_id is None:
        return None, None, None, f"{tag}: no final selected image_id found"

    if final_id < 0 or final_id >= len(target_images):
        return final_id, None, None, f"{tag}: final image_id={final_id} out of range"

    img_path = target_images[final_id]
    pid = proposal_id_from_canvas_path(img_path)
    src = resolve_path(img_path, project_root)

    dst = out_case_dir / f"{tag}_selected_id_{final_id:03d}_pid_{pid}.jpg"
    ok = safe_copy(src, dst)

    msg = None if ok else f"{tag}: missing selected canvas: {src}"
    return final_id, pid, dst, msg


def reconstruct_tournament_rounds(num_candidates, winner_ids, group_size=4):
    """
    根据候选数量和 VLM 每次输出的 image_id，重建 tournament 每一轮。
    返回:
        [
          {
            "round": 1,
            "groups": [[0,1,2,3], [4,5,6,7], ...],
            "winners": [0,6,8],
            "before": [0,1,2,3,4,5,6,7,8],
            "after": [0,6,8],
          },
          ...
        ]
    """
    rounds = []
    current = list(range(num_candidates))
    ptr = 0
    round_idx = 1

    while len(current) > 1 and ptr < len(winner_ids):
        groups = [
            current[i:i + group_size]
            for i in range(0, len(current), group_size)
        ]

        winners = []
        for group in groups:
            if ptr >= len(winner_ids):
                break

            winner = winner_ids[ptr]
            ptr += 1

            # 正常情况下 winner 应该在 group 里
            # 如果不在，也先记录，便于发现 log 或 parser 异常
            winners.append(winner)

        rounds.append({
            "round": round_idx,
            "before": current,
            "groups": groups,
            "winners": winners,
            "after": winners,
        })

        current = winners
        round_idx += 1

    return rounds


def analyze_baseline_correct_candidate_in_global(baseline_block, global_block, iou_threshold=0.25):
    """
    判断 baseline 正确候选在 global 版本中：
    - 是否也被选中
    - 是否进入最后一轮后被改错
    - 是否在前面轮次就被筛掉
    """
    baseline_iou = extract_iou(baseline_block)
    global_iou = extract_iou(global_block)

    baseline_final_id = get_final_selected_id(baseline_block)
    global_final_id = get_final_selected_id(global_block)

    global_target_images = extract_target_images(global_block)
    global_winner_ids = extract_winner_ids(global_block)

    if baseline_iou is None:
        return "unknown: baseline IoU not found"

    if baseline_iou < iou_threshold:
        return "baseline was not correct, so no correct-candidate survival analysis"

    if baseline_final_id is None:
        return "unknown: baseline final selected image_id not found"

    if global_final_id == baseline_final_id:
        return (
            f"baseline correct candidate image_id={baseline_final_id} "
            f"was also selected by global version"
        )

    rounds = reconstruct_tournament_rounds(
        num_candidates=len(global_target_images),
        winner_ids=global_winner_ids,
        group_size=4,
    )

    if not rounds:
        return "unknown: no tournament rounds reconstructed in global log"

    final_round_candidates = rounds[-1]["before"]

    if baseline_final_id in final_round_candidates:
        return (
            f"entered final round but was changed by global/final decision: "
            f"baseline_correct_id={baseline_final_id}, "
            f"final_round_candidates={final_round_candidates}, "
            f"global_selected_id={global_final_id}"
        )

    for r in rounds:
        if baseline_final_id in r["before"] and baseline_final_id not in r["after"]:
            return (
                f"filtered before final round at round {r['round']}: "
                f"baseline_correct_id={baseline_final_id}, "
                f"round_groups={r['groups']}, "
                f"round_winners={r['winners']}, "
                f"global_selected_id={global_final_id}"
            )

    return (
        f"baseline correct candidate image_id={baseline_final_id} "
        f"was not found in reconstructed global path"
    )


def compare_status(baseline_iou, global_iou, threshold=0.25):
    if baseline_iou is None or global_iou is None:
        return "unknown"

    b_ok = baseline_iou >= threshold
    g_ok = global_iou >= threshold

    if (not b_ok) and g_ok:
        return "wrong_to_correct"
    if b_ok and (not g_ok):
        return "correct_to_wrong"
    if b_ok and g_ok:
        return "correct_to_correct"
    return "wrong_to_wrong"

def chinese_round_name(round_idx):
    names = {
        1: "第一轮",
        2: "第二轮",
        3: "第三轮",
        4: "第四轮",
        5: "第五轮",
        6: "第六轮",
        7: "第七轮",
        8: "第八轮",
    }
    return names.get(round_idx, f"第{round_idx}轮")


def format_group(group):
    return "[" + ",".join(str(x) for x in group) + "]"


def format_tournament_rounds(block):
    num_candidates = extract_num_target_images(block)
    winner_ids = extract_winner_ids(block)

    if num_candidates <= 0:
        return "No target candidates found.\n"

    if num_candidates == 1:
        return "Only one candidate, no tournament rounds.\n"

    if not winner_ids:
        return "No VLM winner ids found.\n"

    rounds = reconstruct_tournament_rounds(
        num_candidates=num_candidates,
        winner_ids=winner_ids,
        group_size=4,
    )

    lines = []

    for r in rounds:
        lines.append(f"{chinese_round_name(r['round'])}：")

        for group, winner in zip(r["groups"], r["winners"]):
            group_str = format_group(group)
            lines.append(f"{group_str:<20} -> image_id={winner}")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"

def write_merged_summary(
    case_id,
    baseline_block,
    global_block,
    out_case_dir,
    baseline_final_id,
    baseline_pid,
    global_final_id,
    global_pid,
    copied_global_view_path,
    iou_threshold=0.25,
):
    scene_id = extract_line(global_block, "scene_id") or extract_line(baseline_block, "scene_id")
    caption = extract_line(global_block, "caption") or extract_line(baseline_block, "caption")
    obj_id = extract_line(global_block, "obj_id") or extract_line(baseline_block, "obj_id")
    obj_name = extract_line(global_block, "obj_name") or extract_line(baseline_block, "obj_name")

    baseline_iou = extract_iou(baseline_block)
    global_iou = extract_iou(global_block)

    status = compare_status(baseline_iou, global_iou, threshold=iou_threshold)

    survival_info = analyze_baseline_correct_candidate_in_global(
        baseline_block=baseline_block,
        global_block=global_block,
        iou_threshold=iou_threshold,
    )

    summary_path = out_case_dir / "summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"case: {case_id}\n")
        f.write(f"scene_id: {scene_id}\n")
        f.write(f"obj_id: {obj_id}\n")
        f.write(f"obj_name: {obj_name}\n")
        f.write(f"caption: {caption}\n")
        f.write("\n")

        f.write(f"compare_status@25: {status}\n")
        f.write("\n")

        f.write("[baseline_dynamic_only]\n")
        f.write(f"IoU: {baseline_iou}\n")
        f.write(f"final_selected_image_id: {baseline_final_id}\n")
        f.write(f"final_selected_proposal_id: {baseline_pid}\n")
        f.write("\n")
        f.write("tournament_path:\n")
        f.write(format_tournament_rounds(baseline_block))
        f.write("\n")

        f.write("[global_view]\n")
        f.write(f"IoU: {global_iou}\n")
        f.write(f"final_selected_image_id: {global_final_id}\n")
        f.write(f"final_selected_proposal_id: {global_pid}\n")
        f.write(f"final_global_view_copied_to: {copied_global_view_path}\n")
        f.write("\n")
        f.write("tournament_path:\n")
        f.write(format_tournament_rounds(global_block))
        f.write("\n")

        f.write("[baseline_correct_candidate_survival_in_global]\n")
        f.write(survival_info + "\n")

    print(f"[SUMMARY] {summary_path}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cases",
        nargs="+",
        type=int,
        required=True,
        help="Case ids to collect, e.g. --cases 48 58 108 171 187 197 226",
    )

    parser.add_argument(
        "--baseline-log",
        type=str,
        default=str(DEFAULT_LOG_DIR / "scanrefer_full_20260616_123454.log"),
        help="Baseline dynamic-only log path",
    )

    parser.add_argument(
        "--global-log",
        type=str,
        default=str(DEFAULT_LOG_DIR / "scanrefer_full_20260709_184706.log"),
        help="Global-view log path",
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=str(DEFAULT_PROJECT_ROOT),
        help="Project root used to resolve relative image paths like ../data/...",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_PROJECT_ROOT / "selected_case_compare"),
        help="Output folder",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    baseline_log_path = Path(args.baseline_log).resolve()
    global_log_path = Path(args.global_log).resolve()
    out_root = Path(args.out).resolve()

    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Baseline log: {baseline_log_path}")
    print(f"Global log:   {global_log_path}")
    print(f"Output root:  {out_root}")
    print(f"Cases:        {args.cases}")
    print()

    baseline_text = baseline_log_path.read_text(encoding="utf-8", errors="ignore")
    global_text = global_log_path.read_text(encoding="utf-8", errors="ignore")

    baseline_cases = split_cases(baseline_text)
    global_cases = split_cases(global_text)

    for case_id in args.cases:
        print("=" * 80)
        print(f"Collecting Case {case_id}")
        print("=" * 80)

        baseline_block = baseline_cases.get(case_id)
        global_block = global_cases.get(case_id)

        if baseline_block is None:
            print(f"[WARN] Case {case_id} not found in baseline log")
            continue

        if global_block is None:
            print(f"[WARN] Case {case_id} not found in global log")
            continue

        scene_id = extract_line(global_block, "scene_id") or extract_line(baseline_block, "scene_id")
        out_case_dir = out_root / f"case_{case_id:03d}_{scene_id}"
        out_case_dir.mkdir(parents=True, exist_ok=True)

        # 1. copy baseline final selected canvas
        baseline_final_id, baseline_pid, baseline_dst, baseline_msg = copy_final_selected_canvas(
            block=baseline_block,
            tag="baseline",
            out_case_dir=out_case_dir,
            project_root=project_root,
        )
        if baseline_msg:
            print(f"[WARN] {baseline_msg}")

        # 2. copy global final selected canvas
        global_final_id, global_pid, global_dst, global_msg = copy_final_selected_canvas(
            block=global_block,
            tag="global",
            out_case_dir=out_case_dir,
            project_root=project_root,
        )
        if global_msg:
            print(f"[WARN] {global_msg}")

        # 3. copy final global view
        final_global_view = extract_global_view(global_block)
        copied_global_view_path = None

        if final_global_view:
            src = resolve_path(final_global_view, project_root)
            dst = out_case_dir / "final_global_view.jpg"
            ok = safe_copy(src, dst)
            if ok:
                copied_global_view_path = dst
        else:
            print(f"[WARN] Case {case_id}: no FinalGlobalView found in global log")

        # 4. write one merged summary.txt
        write_merged_summary(
            case_id=case_id,
            baseline_block=baseline_block,
            global_block=global_block,
            out_case_dir=out_case_dir,
            baseline_final_id=baseline_final_id,
            baseline_pid=baseline_pid,
            global_final_id=global_final_id,
            global_pid=global_pid,
            copied_global_view_path=copied_global_view_path,
        )

    print("\nDone.")
    print(f"Output folder: {out_root}")


if __name__ == "__main__":
    main()