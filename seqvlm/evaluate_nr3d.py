'''
推理主流程：Nr3D
'''

import os
import random
import argparse
import json
import numpy as np

from utils import *
from seqvlm.adaptive_predictor_nr3d import AdpativePredictor


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='seqvlm nr3d')

    # basic args
    parser.add_argument('--data_path', type=str, default='../data/nr3d_250.json')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--vlm_model', type=str, default='qwen')
    parser.add_argument('--max_retry', type=int, default=3)
    parser.add_argument('--max_batch_size', type=int, default=4)
    parser.add_argument('--max_vlm_props', type=int, default=40)
    parser.add_argument('--max_samples', type=int, default=None, help='limit eval samples for smoke test')
    parser.add_argument('--seed', type=int, default=42)

    # anchor-aware args
    parser.add_argument('--use_anchor_aware', action='store_true')
    parser.add_argument('--max_anchor_per_type', type=int, default=5)
    parser.add_argument('--seg_conf_score', type=float, default=0.0)

    # dynamic canvas args
    parser.add_argument('--use_dynamic_canvas', action='store_true')
    parser.add_argument('--crop_image_root', type=str, default='../data/crop_images_nr3d')
    parser.add_argument('--crop_pool_meta_root', type=str, default='../data/crop_pool_meta_nr3d')
    parser.add_argument('--view_meta_root', type=str, default='../data/view_meta_nr3d')
    parser.add_argument('--posed_image_root', type=str, default='../data/posed_images_rgb_pose')
    parser.add_argument('--dynamic_canvas_root', type=str, default='../data/dynamic_canvas_nr3d')
    parser.add_argument('--canvas_k', type=int, default=5)
    parser.add_argument('--num_appearance_views', type=int, default=2)
    parser.add_argument('--num_relation_views', type=int, default=2)
    parser.add_argument('--use_global_context', action='store_true')

    args = parser.parse_args()
    set_seed(args.seed)

    print("[Config]")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))

    metrics_logger, full_log_path, metrics_log_path = setup_run_logging('nr3d')

    with open(args.data_path, 'r') as f:
        eval_data = json.load(f)

    correct_25 = 0
    correct_50 = 0
    correct_easy = 0
    correct_dep = 0

    easy_total = 0
    dep_total = 0
    total = 0
    vlm_total = 0
    except_total = 0

    eps = 10 ** -6

    vlm_configs = {
        'image_path': args.image_path,
        'vlm_model': args.vlm_model,
        'max_retry': args.max_retry,
        'max_batch_size': args.max_batch_size,
        'max_vlm_props': args.max_vlm_props,

        # anchor-aware
        'use_anchor_aware': args.use_anchor_aware,
        'max_anchor_per_type': args.max_anchor_per_type,
        'seg_conf_score': args.seg_conf_score,
        'query_parse_cache_dir': '../data/cache/query_parse_nr3d',

        # dynamic canvas
        'use_dynamic_canvas': args.use_dynamic_canvas,
        'crop_image_root': args.crop_image_root,
        'crop_pool_meta_root': args.crop_pool_meta_root,
        'view_meta_root': args.view_meta_root,
        'posed_image_root': args.posed_image_root,
        'dynamic_canvas_root': args.dynamic_canvas_root,
        'canvas_k': args.canvas_k,
        'num_appearance_views': args.num_appearance_views,
        'num_relation_views': args.num_relation_views,
        'use_global_context': args.use_global_context,
    }

    predictor = AdpativePredictor(**vlm_configs)

    if args.max_samples is not None:
        eval_data = eval_data[:args.max_samples]

    for i, task in enumerate(eval_data):
        scene_id, obj_id, caption, prog_str, is_easy, is_dep, obj_name = task.values()

        print('Case:', i)
        print('scene_id:', scene_id)
        print('caption:', caption)
        print('obj_id:', obj_id)
        print('obj_name:', obj_name)

        obj_ids, obj_labels, obj_locs = load_pc(scene_id)

        try:
            index = list(obj_ids).index(int(obj_id))
        except ValueError:
            print(
                f"[EXCEPT_CASE] GT obj_id={obj_id} not found "
                f"in scene={scene_id}"
            )
            total += 1
            except_total += 1
            continue

        target_box = obj_locs[index]

        total += 1

        if is_easy:
            easy_total += 1
        if is_dep:
            dep_total += 1

        pred_box, use_vlm = predictor.execute(scene_id, obj_name, caption, prog_str)

        if use_vlm:
            vlm_total += 1

        if pred_box is None:
            except_total += 1
            print(
                f"[EXCEPT_CASE] case={i}, scene_id={scene_id}, "
                f"obj_id={obj_id}, obj_name={obj_name}, "
                f"use_vlm={use_vlm}, caption={caption}"
            )
        else:
            iou = calc_iou(pred_box, target_box)
            print(f'IoU: {iou:.2f}')

            if iou >= 0.25:
                correct_25 += 1
                if is_easy:
                    correct_easy += 1
                if is_dep:
                    correct_dep += 1

            if iou >= 0.5:
                correct_50 += 1

        accuracy_msgs = [
            'Overall@25: {:.3f}'.format(correct_25 / total),
            'Overall@50: {:.3f}'.format(correct_50 / total),
            'Easy: {:.3f}'.format(correct_easy / (easy_total + eps)),
            'Hard: {:.3f}'.format((correct_25 - correct_easy) / (total - easy_total + eps)),
            'VD: {:.3f}'.format(correct_dep / (dep_total + eps)),
            'VID: {:.3f}'.format((correct_25 - correct_dep) / (total - dep_total + eps)),
            'Easy Ratio: {} / {}'.format(correct_easy, easy_total),
            'Hard Ratio: {} / {}'.format(correct_25 - correct_easy, total - easy_total),
            'VD Ratio: {} / {}'.format(correct_dep, dep_total),
            'VID Ratio: {} / {}'.format(correct_25 - correct_dep, total - dep_total),
            'Except Ratio: {} / {}'.format(except_total, total),
            'VLM Usage Ratio: {} / {}'.format(vlm_total, total),
            '\n'
        ]

        print('\n'.join(accuracy_msgs))

        if (i + 1) % 10 == 0:
            metrics_logger.info(f'--- Case {i} ---')
            for msg in accuracy_msgs:
                if msg.strip():
                    metrics_logger.info(msg)