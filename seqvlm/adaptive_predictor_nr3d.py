'''
候选框生成/筛选：NR3D
'''

import json
import os
import pickle

from objprint import op
from colorama import Fore, init, Style
init(autoreset=True)

from prompts.prompt import *
from utils import *
from api import invoke_api

from seqvlm.feat_handler_nr3d import VisualFeatHandler
from seqvlm.query_parser import QueryParser
from seqvlm.dynamic_view_selector import DynamicViewSelector

from seqvlm.anchor_utils_nr3d import (
    build_anchor_candidates_nr3d,
    summarize_anchors_for_prompt,
    build_target_candidate_meta_nr3d,
    summarize_candidate_batch_for_prompt_nr3d,
)


'''
加载3D特征： feats_3d.pkl
'''
with open('../data/feats_3d.pkl', 'rb') as f:
    feats = pickle.load(f)


def load_pc(scan_id):
    obj_ids = feats[scan_id]['obj_ids']
    inst_locs = feats[scan_id]['inst_locs']
    center = feats[scan_id]['center']
    obj_embeds = feats[scan_id]['obj_embeds']
    return obj_ids, inst_locs, center, obj_embeds


class AdpativePredictor:
    def __init__(self, **kwargs):
        self.image_path = kwargs.get('image_path')
        self.max_retry = kwargs.get('max_retry')
        self.vlm_model = kwargs.get('vlm_model')
        self.max_batch_size = kwargs.get('max_batch_size')
        self.max_vlm_props = kwargs.get('max_vlm_props')
        self.handler = VisualFeatHandler.get_instance()

        # Anchor-aware
        self.use_anchor_aware = kwargs.get('use_anchor_aware', True)
        self.max_anchor_per_type = kwargs.get('max_anchor_per_type', 5)
        self.seg_conf_score = kwargs.get('seg_conf_score', 0.0)

        self.query_parser = QueryParser(
            vlm_model=self.vlm_model,
            cache_dir=kwargs.get(
                'query_parse_cache_dir',
                '../data/cache/query_parse_nr3d'
            ),
            max_retry=self.max_retry,
        )

        # Dynamic canvas
        self.use_dynamic_canvas = kwargs.get('use_dynamic_canvas', False)
        self.dynamic_canvas_root = kwargs.get(
            'dynamic_canvas_root',
            '../data/dynamic_canvas_nr3d'
        )

        if self.use_dynamic_canvas:
            self.dynamic_view_selector = DynamicViewSelector(
                crop_image_root=kwargs.get(
                    'crop_image_root',
                    '../data/crop_images_nr3d'
                ),
                crop_pool_meta_root=kwargs.get(
                    'crop_pool_meta_root',
                    '../data/crop_pool_meta_nr3d'
                ),
                view_meta_root=kwargs.get(
                    'view_meta_root',
                    '../data/view_meta_nr3d'
                ),
                posed_image_root=kwargs.get(
                    'posed_image_root',
                    '../data/posed_images_rgb_pose'
                ),
                canvas_root=self.dynamic_canvas_root,
                canvas_k=kwargs.get('canvas_k', 5),
                num_appearance_views=kwargs.get('num_appearance_views', 2),
                num_relation_views=kwargs.get('num_relation_views', 2),
                use_global_context=kwargs.get('use_global_context', True),
            )
        else:
            self.dynamic_view_selector = None

    def execute(self, scene_id, obj_name, caption, prog_str):
        '''
        NR3D 候选框筛选逻辑：
        1. 从 feats_3d.pkl 读取 obj_ids / inst_locs / obj_embeds
        2. 根据 obj_name 预测 target class
        3. 用 query_parser 解析 target / anchor / relation
        4. 根据 NR3D local_index 构建 anchor candidates
        5. 根据 query 动态生成 query-specific canvas
        6. 如果 dynamic canvas 失败，fallback 到 fixed canvas
        7. VLM 在候选 canvas 中选择最终目标
        '''
        obj_ids, ins_locs, center, obj_embeds = load_pc(scene_id)

        pred_cls, pred_class_list = self.handler.predict_obj_class(
            obj_name,
            obj_embeds
        )

        # 1. Query parsing
        parsed_query = None
        anchor_infos = []
        anchor_summary = "Anchor-aware disabled."

        if self.use_anchor_aware:
            parsed_query = self.query_parser.parse(
                query=caption,
                fallback_target=obj_name,
            )

            anchor_infos = build_anchor_candidates_nr3d(
                anchors=parsed_query.get("anchors", []),
                obj_ids=obj_ids,
                pred_class_list=pred_class_list,
                ins_locs=ins_locs,
                obj_embeds=obj_embeds,
                handler=self.handler,
                seg_conf_score=self.seg_conf_score,
                max_anchor_per_type=self.max_anchor_per_type,
            )

            anchor_summary = summarize_anchors_for_prompt(anchor_infos)

            print("Parsed Query:")
            op(parsed_query)
            print("Anchor Infos:")
            op(anchor_infos[:10])

        # 2. Build target candidate canvases
        target_local_indices = []
        target_obj_ids = []
        prop_images = []

        for local_idx in range(len(obj_ids)):
            if pred_class_list[local_idx] != pred_cls:
                continue

            obj_id = int(obj_ids[local_idx])

            # 关键：
            # NR3D dynamic canvas / crop_images_nr3d / view_meta_nr3d / crop_pool_meta_nr3d
            # 全部使用 local_idx 作为 proposal_id。
            dynamic_proposal_id = int(local_idx)

            canvas = None

            if self.use_dynamic_canvas:
                canvas = self.dynamic_view_selector.build_query_specific_canvas(
                    scene_id=scene_id,
                    target_proposal_id=dynamic_proposal_id,
                    query=caption,
                    parsed_query=parsed_query,
                    anchor_infos=anchor_infos,
                    candidate_meta={
                        "proposal_id": dynamic_proposal_id,
                        "obj_id": obj_id,
                        "local_index": int(local_idx),
                        "label": pred_class_list[local_idx],
                        "score": 1.0,
                        "loc": ins_locs[local_idx],
                    },
                )

                # fallback：NR3D fixed canvas 通常也是 local_idx 编号
                if canvas is None or not os.path.exists(canvas):
                    fallback_canvas = os.path.join(
                        self.image_path,
                        scene_id,
                        str(dynamic_proposal_id),
                        "canvas.jpg",
                    )

                    if os.path.exists(fallback_canvas):
                        print(
                            f"[Fallback] use fixed canvas for "
                            f"scene={scene_id}, local_idx={dynamic_proposal_id}, obj_id={obj_id}"
                        )
                        canvas = fallback_canvas

            else:
                canvas = os.path.join(
                    self.image_path,
                    scene_id,
                    str(dynamic_proposal_id),
                    "canvas.jpg",
                )

            if canvas is not None and os.path.exists(canvas):
                target_local_indices.append(local_idx)
                target_obj_ids.append(obj_id)
                prop_images.append(canvas)

        n_props = len(prop_images)
        print('Target Prop Images:', n_props, prop_images[:20])

        if n_props == 0:
            return None, False

        # 单候选无需调用 VLM
        if n_props == 1:
            return ins_locs[target_local_indices[0]], False

        # 3. Truncate if too many proposals
        if self.max_vlm_props is not None and n_props > self.max_vlm_props:
            print(f"[Truncate] Target proposals: {n_props} -> {self.max_vlm_props}")

            # NR3D 没有 segmentation score，这里用 local_idx 稳定截断，保证可复现
            ranked = sorted(
                range(len(target_local_indices)),
                key=lambda k: int(target_local_indices[k])
            )[: self.max_vlm_props]

            target_local_indices = [target_local_indices[k] for k in ranked]
            target_obj_ids = [target_obj_ids[k] for k in ranked]
            prop_images = [prop_images[k] for k in ranked]
            n_props = len(prop_images)

        candidate_metas = build_target_candidate_meta_nr3d(
            target_local_indices=target_local_indices,
            target_obj_ids=target_obj_ids,
            pred_class_list=pred_class_list,
            ins_locs=ins_locs,
            scores=[1.0 for _ in target_local_indices],
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
            return ins_locs[target_local_indices[pred]], True

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

        base64Images = [
            encode_image_to_base64(image)
            for image in prop_images
        ]

        # remain_props: [(global_candidate_id, base64_image), ...]
        remain_props = [
            (i, image)
            for i, image in enumerate(base64Images)
        ]

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
            candidate_summary = summarize_candidate_batch_for_prompt_nr3d(
                batch_items=images,
                candidate_metas=candidate_metas or [],
                anchor_infos=anchor_infos or [],
            )

            if self.use_dynamic_canvas:
                user_prompt = build_dynamic_anchor_aware_user_prompt(
                    query=caption,
                    parsed_query=json.dumps(
                        parsed_query or {},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    anchor_summary=anchor_summary or "",
                    candidate_summary=candidate_summary,
                    n_images=n_images,
                )
                system_prompt = DYNAMIC_ANCHOR_AWARE_SYSTEM_PROMPT

            else:
                user_prompt = build_anchor_aware_user_prompt(
                    query=caption,
                    parsed_query=json.dumps(
                        parsed_query or {},
                        ensure_ascii=False,
                        indent=2,
                    ),
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
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
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

                # Anchor-aware / dynamic prompt:
                # image_id 是 prop_images 的全局候选 id，不是当前 batch 内部下标。
                if isinstance(image_id, int) and image_id in valid_global_ids:
                    return image_id

                guide_prompt = (
                    f"Invalid image_id={image_id}. "
                    f"You must choose one image_id from this list: {valid_global_ids}. "
                    f"Return ONLY valid JSON."
                )

            except Exception as e:
                print(Fore.RED + f"Except: {e}" + Style.RESET_ALL)
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