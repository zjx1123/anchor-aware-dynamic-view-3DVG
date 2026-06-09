'''
【用途】：给一个 scene_id + target_id + anchor_id + query，单独生成 canvas，验证红框/蓝框是否正确。
'''

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath("."))

from seqvlm.dynamic_view_selector import DynamicViewSelector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--target_id", type=int, required=True)
    parser.add_argument("--anchor_id", type=int, required=True)
    parser.add_argument("--query", type=str, default="debug query")

    parser.add_argument("--crop_image_root", default="data/crop_images")
    parser.add_argument("--crop_pool_meta_root", default="data/crop_pool_meta_scanrefer")
    parser.add_argument("--view_meta_root", default="data/view_meta_scanrefer")
    parser.add_argument("--posed_image_root", default="data/posed_images_rgb_pose")
    parser.add_argument("--canvas_root", default="data/debug_dynamic_canvas")

    args = parser.parse_args()

    selector = DynamicViewSelector(
        crop_image_root=args.crop_image_root,
        crop_pool_meta_root=args.crop_pool_meta_root,
        view_meta_root=args.view_meta_root,
        posed_image_root=args.posed_image_root,
        canvas_root=args.canvas_root,
        canvas_k=5,
    )

    anchor_infos = [
        {
            "proposal_id": args.anchor_id,
            "score": 1.0,
            "matched_class": "debug_anchor",
            "relation_to_target": "debug_relation",
            "attributes": [],
            "loc": [0, 0, 0, 0, 0, 0],
        }
    ]

    canvas = selector.build_query_specific_canvas(
        scene_id=args.scene_id,
        target_proposal_id=args.target_id,
        query=args.query,
        parsed_query={},
        anchor_infos=anchor_infos,
    )

    print("canvas:", canvas)


if __name__ == "__main__":
    main()