'''
候选框生成/筛选： ScanRefer
'''
import glob
import json
import os
import random
from objprint import op
from colorama import Fore, init
init(autoreset=True)

from prompts.prompt import *
from utils import *
from api import invoke_api
from seqvlm.feat_handler import VisualFeatHandler

from seqvlm.query_parser import QueryParser
from seqvlm.anchor_utils import (
    build_anchor_candidates,
    summarize_anchors_for_prompt,
    build_target_candidate_meta,
    summarize_candidate_batch_for_prompt,
)



class AdpativePredictor:
    
    def __init__(self, **kwargs):        
        self.image_path = kwargs.get('image_path')
        self.max_retry = kwargs.get('max_retry')
        self.vlm_model = kwargs.get('vlm_model')
        self.max_batch_size = kwargs.get('max_batch_size')
        self.max_vlm_props = kwargs.get('max_vlm_props')
        
        self.handler = VisualFeatHandler.get_instance()

        # Anchor-aware 新增
        self.use_anchor_aware = kwargs.get('use_anchor_aware', True)
        self.max_anchor_per_type = kwargs.get('max_anchor_per_type', 5)
        self.seg_conf_score = kwargs.get('seg_conf_score', 0.2)

        self.query_parser = QueryParser(
            vlm_model=self.vlm_model,
            cache_dir=kwargs.get(
                'query_parse_cache_dir',
                '../data/cache/query_parse_scanrefer'
            ),
            max_retry=self.max_retry,
        )

    
    def execute(self, scene_id, obj_name, caption, prog_str):
        ins_labels, ins_locs, ins_scores, _ = load_seg_inst(scene_id)

        # 1. 原 SeqVLM target class 预测仍保留
        pred_cls = self.handler.predict_obj_class(obj_name, ins_labels)

        # 2. Anchor-aware: 解析 query
        parsed_query = None
        anchor_infos = []
        anchor_summary = "Anchor-aware disabled."

        if self.use_anchor_aware:
            parsed_query = self.query_parser.parse(
                query=caption,
                fallback_target=obj_name,
            )

            anchor_infos = build_anchor_candidates(
                anchors=parsed_query.get("anchors", []),
                ins_labels=ins_labels,
                ins_locs=ins_locs,
                ins_scores=ins_scores,
                handler=self.handler,
                image_path=self.image_path,
                scene_id=scene_id,
                seg_conf_score=self.seg_conf_score,
                max_anchor_per_type=self.max_anchor_per_type,
            )

            anchor_summary = summarize_anchors_for_prompt(anchor_infos)

            print("Parsed Query:")
            op(parsed_query)
            print("Anchor Infos:")
            op(anchor_infos[:10])

        # 3. target proposals 仍然只保留 target class
        index = []
        prop_images = []

        for i, label in enumerate(ins_labels):
            if ins_scores[i] > self.seg_conf_score and label == pred_cls:
                canvas = os.path.join(self.image_path, scene_id, str(i), 'canvas.jpg')
                if os.path.exists(canvas):
                    index.append(i)
                    prop_images.append(canvas)

        n_props = len(prop_images)
        print('Target Prop Images:', n_props, prop_images[:20])

        if n_props == 0:
            return None, False

        if n_props <= self.max_vlm_props:
            candidate_metas = build_target_candidate_meta(
                target_indices=index,
                ins_labels=ins_labels,
                ins_locs=ins_locs,
                ins_scores=ins_scores,
            )

            pred = self.predict(
                prop_images=prop_images,
                caption=caption,
                parsed_query=parsed_query,
                anchor_infos=anchor_infos,
                anchor_summary=anchor_summary,
                candidate_metas=candidate_metas,
            )

            if pred is not None:
                return ins_locs[index[pred]], True

        return None, False

    def predict(
        self,
        prop_images,
        caption,
        parsed_query=None,
        anchor_infos=None,
        anchor_summary=None,
        candidate_metas=None,
    ):
        max_batch_size = self.max_batch_size

        base64Images = [encode_image_to_base64(image) for image in prop_images]

        # remain_props: [(global_candidate_id, base64_image), ...]
        remain_props = [(i, image) for i, image in enumerate(base64Images)]

        while len(remain_props) > 1:
            prop_image_groups = []
            for i in range(0, len(remain_props), max_batch_size):
                g = remain_props[i:i + max_batch_size]
                prop_image_groups.append(g)

            next_round = []

            for images in prop_image_groups:
                selected_global_id = self.select_with_retry(
                    images=images,
                    caption=caption,
                    parsed_query=parsed_query,
                    anchor_infos=anchor_infos or [],
                    anchor_summary=anchor_summary or "",
                    candidate_metas=candidate_metas or [],
                )

                if selected_global_id != -1:
                    # 从当前 batch 中找回对应 item
                    selected_item = None
                    for item in images:
                        if item[0] == selected_global_id:
                            selected_item = item
                            break

                    if selected_item is not None:
                        next_round.append(selected_item)

            remain_props = next_round

        return remain_props[0][0] if remain_props else None

    def select_with_retry(
        self,
        images,
        caption,
        parsed_query=None,
        anchor_infos=None,
        anchor_summary=None,
        candidate_metas=None,
    ):
        n_images = len(images)
        assert n_images > 0

        # images: [(global_candidate_id, base64_image), ...]
        valid_global_ids = [x[0] for x in images]

        if self.use_anchor_aware:
            candidate_summary = summarize_candidate_batch_for_prompt(
                batch_items=images,
                candidate_metas=candidate_metas or [],
                anchor_infos=anchor_infos or [],
            )

            user_prompt = build_anchor_aware_user_prompt(
                query=caption,
                parsed_query=json.dumps(parsed_query or {}, ensure_ascii=False, indent=2),
                anchor_summary=anchor_summary or "",
                candidate_summary=candidate_summary,
                n_images=n_images,
            )

            system_prompt = ANCHOR_AWARE_SYSTEM_PROMPT
        else:
            user_prompt = USER_PROMPT.format(
                query=caption,
                n_images=n_images,
            )
            system_prompt = SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    *map(
                        lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{x[1]}",
                                "detail": "high",
                            },
                        },
                        images,
                    ),
                ],
            },
        ]

        retry = 0
        output = {"answer": ""}

        while retry < self.max_retry:
            try:
                output = invoke_api(self.vlm_model, messages)
                print('Vision Lang Model Output:')
                op(output)

                answer = json.loads(output['answer'])
                image_id = answer['image_id']

                # Anchor-aware: image_id 是全局候选 id
                if isinstance(image_id, int) and image_id in valid_global_ids:
                    return image_id

                guide_prompt = (
                    f"Invalid image_id={image_id}. "
                    f"You must choose one image_id from this list: {valid_global_ids}. "
                    f"Return ONLY valid JSON."
                )

            except Exception as e:
                print(Fore.RED + f'Except: {e}')
                guide_prompt = WRONG_FORMAT_PROMPT

            messages.append({
                'role': 'assistant',
                'content': output.get('answer', ''),
            })
            messages.append({
                "role": "user",
                "content": guide_prompt,
            })

            retry += 1

        return -1
