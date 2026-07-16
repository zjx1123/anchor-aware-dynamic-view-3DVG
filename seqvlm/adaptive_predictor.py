'''
候选框生成/筛选： ScanRefer
'''
import glob
import json
import os
import random
from objprint import op
from colorama import Fore, init, Style
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

from seqvlm.dynamic_view_selector import DynamicViewSelector
from seqvlm.final_global_view_builder import FinalGlobalViewBuilder

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
        self.seg_conf_score = kwargs.get('seg_conf_score', 0.2)

        self.query_parser = QueryParser(
            vlm_model=self.vlm_model,
            cache_dir=kwargs.get(
                'query_parse_cache_dir',
                '../data/cache/query_parse_scanrefer'
            ),
            max_retry=self.max_retry,
        )

        # Dynamic canvas 新增
        self.use_dynamic_canvas = kwargs.get('use_dynamic_canvas', False)
        self.dynamic_canvas_root = kwargs.get(
            'dynamic_canvas_root',
            '../data/dynamic_canvas_scanrefer'
        )

        if self.use_dynamic_canvas:
            self.dynamic_view_selector = DynamicViewSelector(
                crop_image_root=kwargs.get(
                    'crop_image_root',
                    '../data/crop_images'
                ),
                crop_pool_meta_root=kwargs.get(
                    'crop_pool_meta_root',
                    '../data/crop_pool_meta_scanrefer'
                ),
                view_meta_root=kwargs.get(
                    'view_meta_root',
                    '../data/view_meta_scanrefer'
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

        # Final-round global auxiliary view
        self.use_final_global_view = kwargs.get("use_final_global_view", False)

        # 如果 use_final_global_view=True，默认启用 gate。
        # 如果之后想做 ablation，可以在 evaluate.py 里传 use_final_global_gate=False。
        self.use_final_global_gate = kwargs.get("use_final_global_gate", True)

        self.final_global_view_builder = None

        if self.use_final_global_view:
            self.final_global_view_builder = FinalGlobalViewBuilder(
                rendered_root=kwargs.get(
                    "global_rendered_root",
                    "../data/global_rendered_views_scanrefer",
                ),
                out_root=kwargs.get(
                    "final_global_view_root",
                    "../data/final_global_aux_scanrefer",
                ),
                max_anchors=kwargs.get("max_final_global_anchors", 3),
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

        # 3. Dynamic canvas: 如果启用，则动态生成 canvas
        index = []
        prop_images = []

        for i, label in enumerate(ins_labels):
            if ins_scores[i] > self.seg_conf_score and label == pred_cls:

                if self.use_dynamic_canvas:
                    # 为当前 query-proposal pair 动态生成 canvas
                    canvas = self.dynamic_view_selector.build_query_specific_canvas(
                        scene_id=scene_id,
                        target_proposal_id=i,
                        query=caption,
                        parsed_query=parsed_query,
                        anchor_infos=anchor_infos,
                        candidate_meta={
                            "proposal_id": i,
                            "label": label,
                            "score": float(ins_scores[i]),
                            "loc": ins_locs[i],
                        },
                    )

                    # 关键 fallback：
                    # dynamic canvas 失败时，回退到原始固定 canvas
                    if canvas is None or not os.path.exists(canvas):
                        fallback_canvas = os.path.join(
                            self.image_path,
                            scene_id,
                            str(i),
                            "canvas.jpg",
                        )
                        if os.path.exists(fallback_canvas):
                            print(f"[Fallback] use fixed canvas for scene={scene_id}, proposal={i}")
                            canvas = fallback_canvas
                else:
                    canvas = os.path.join(self.image_path, scene_id, str(i), "canvas.jpg")

                if canvas is not None and os.path.exists(canvas):
                    index.append(i)
                    prop_images.append(canvas)

        n_props = len(prop_images)
        print('Target Prop Images:', n_props, prop_images[:20])

        if n_props == 0:
            return None, False

        # =========================
        # Important fix:
        # If candidate proposals are more than max_vlm_props,
        # do NOT directly fail. Truncate them before VLM reasoning.
        # =========================
        if self.max_vlm_props is not None and n_props > self.max_vlm_props:
            print(f"[Truncate] Target proposals: {n_props} -> {self.max_vlm_props}")

            # 第一版先按 proposal score 排序截断
            ranked = sorted(
                range(len(index)),
                key=lambda k: float(ins_scores[index[k]]),
                reverse=True,
            )[: self.max_vlm_props]

            index = [index[k] for k in ranked]
            prop_images = [prop_images[k] for k in ranked]
            n_props = len(prop_images)

        candidate_metas = build_target_candidate_meta(
            target_indices=index,
            ins_labels=ins_labels,
            ins_locs=ins_locs,
            ins_scores=ins_scores,
        )

        pred = self.predict(
            scene_id=scene_id,
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
        scene_id,
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

        # image_id -> candidate meta
        meta_by_image_id = {
            int(m["local_image_id"]): m
            for m in (candidate_metas or [])
        }

        while len(remain_props) > 1:
            prop_image_groups = []

            for i in range(0, len(remain_props), max_batch_size):
                g = remain_props[i:i + max_batch_size]
                prop_image_groups.append(g)

            # If there is only one group in this round,
            # this round will select the final answer.
            is_final_round = len(prop_image_groups) == 1

            next_round = []

            for images in prop_image_groups:
                selected_global_id = -1

                # ==========================================================
                # New final-round gated global logic:
                #
                # Step 1. In final round, first run local-only gate.
                # Step 2. The VLM decides whether global view is necessary.
                # Step 3. Only if need_global=True, build and use global view.
                # Step 4. If global decision fails, fall back to local result.
                # ==========================================================
                if (
                    is_final_round
                    and self.use_final_global_view
                    and self.use_final_global_gate
                    and self.final_global_view_builder is not None
                ):
                    # ---------- Step 1: local-only final gate ----------
                    local_selected_id, local_answer = self.select_with_retry(
                        images=images,
                        caption=caption,
                        parsed_query=parsed_query,
                        anchor_infos=anchor_infos or [],
                        anchor_summary=anchor_summary or "",
                        candidate_metas=candidate_metas or [],
                        final_global_view_path=None,  # critical: no global image in gate stage
                        prompt_mode="final_local_gate",
                        return_answer=True,
                    )

                    need_global, gate_reason = self.parse_need_global(local_answer)

                    print(
                        f"[FinalGlobalGate] scene={scene_id}, "
                        f"need_global={need_global}, "
                        f"local_selected={local_selected_id}, "
                        f"reason={gate_reason}"
                    )

                    # If local gate failed, do not directly fail.
                    # Fall back to normal local decision.
                    if local_selected_id == -1:
                        print(
                            f"[FinalGlobalGateFallback] scene={scene_id}, "
                            f"local gate failed, use normal local selection."
                        )

                        selected_global_id = self.select_with_retry(
                            images=images,
                            caption=caption,
                            parsed_query=parsed_query,
                            anchor_infos=anchor_infos or [],
                            anchor_summary=anchor_summary or "",
                            candidate_metas=candidate_metas or [],
                            final_global_view_path=None,
                            prompt_mode="normal",
                        )

                    elif not need_global:
                        # ---------- Step 2a: use local decision directly ----------
                        print(
                            f"[FinalGlobalSkipped] scene={scene_id}, "
                            f"selected={local_selected_id}"
                        )
                        selected_global_id = local_selected_id

                    else:
                        # ---------- Step 2b: build global view only when needed ----------
                        final_candidates = []

                        for image_id, _ in images:
                            meta = meta_by_image_id.get(int(image_id))
                            if meta is None:
                                continue

                            final_candidates.append({
                                "image_id": int(image_id),
                                "proposal_id": int(meta["proposal_id"]),
                            })

                        final_global_view_path = self.final_global_view_builder.build(
                            scene_id=scene_id,
                            query=caption,
                            final_candidates=final_candidates,
                            anchor_infos=anchor_infos or [],
                            parsed_query=parsed_query,
                        )

                        if final_global_view_path is not None:
                            print(
                                f"[FinalGlobalView] scene={scene_id}, "
                                f"path={final_global_view_path}"
                            )

                        # ---------- Step 3: final decision with global view ----------
                        selected_global_id = self.select_with_retry(
                            images=images,
                            caption=caption,
                            parsed_query=parsed_query,
                            anchor_infos=anchor_infos or [],
                            anchor_summary=anchor_summary or "",
                            candidate_metas=candidate_metas or [],
                            final_global_view_path=final_global_view_path,
                            prompt_mode="final_global_decision",
                            gate_reason=gate_reason,
                        )

                        # ---------- Step 4: fallback to local gate result ----------
                        if selected_global_id == -1:
                            print(
                                f"[FinalGlobalFallback] scene={scene_id}, "
                                f"global decision failed, use local_selected={local_selected_id}"
                            )
                            selected_global_id = local_selected_id

                else:
                    # ==========================================================
                    # Original logic:
                    # - non-final rounds: normal dynamic canvas
                    # - final round with gate disabled: old always-on global
                    # - use_final_global_view=False: normal dynamic canvas
                    # ==========================================================
                    final_global_view_path = None

                    if (
                        is_final_round
                        and self.use_final_global_view
                        and not self.use_final_global_gate
                        and self.final_global_view_builder is not None
                    ):
                        final_candidates = []

                        for image_id, _ in images:
                            meta = meta_by_image_id.get(int(image_id))
                            if meta is None:
                                continue

                            final_candidates.append({
                                "image_id": int(image_id),
                                "proposal_id": int(meta["proposal_id"]),
                            })

                        final_global_view_path = self.final_global_view_builder.build(
                            scene_id=scene_id,
                            query=caption,
                            final_candidates=final_candidates,
                            anchor_infos=anchor_infos or [],
                            parsed_query=parsed_query,
                        )

                        if final_global_view_path is not None:
                            print(
                                f"[FinalGlobalView] scene={scene_id}, "
                                f"path={final_global_view_path}"
                            )

                    selected_global_id = self.select_with_retry(
                        images=images,
                        caption=caption,
                        parsed_query=parsed_query,
                        anchor_infos=anchor_infos or [],
                        anchor_summary=anchor_summary or "",
                        candidate_metas=candidate_metas or [],
                        final_global_view_path=final_global_view_path,
                        prompt_mode="normal",
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

    def parse_need_global(self, answer):
        """
        FINAL_LOCAL_GATE_PROMPT output format:
        {
            "process": "...",
            "need_global": true/false,
            "global_need_reason": "...",
            "image_id": 0
        }

        If parsing fails, default to need_global=False to avoid unnecessary global-view harm.
        """
        need_global = False
        reason = ""

        try:
            if isinstance(answer, dict):
                raw_need = answer.get("need_global", False)
                reason = answer.get("global_need_reason", "")

                if isinstance(raw_need, bool):
                    need_global = raw_need
                elif isinstance(raw_need, str):
                    need_global = raw_need.strip().lower() in ["true", "yes", "1"]
                elif isinstance(raw_need, int):
                    need_global = bool(raw_need)

        except Exception:
            need_global = False
            reason = ""

        return need_global, reason

    def select_with_retry(
        self,
        images,
        caption,
        parsed_query=None,
        anchor_infos=None,
        anchor_summary=None,
        candidate_metas=None,
        final_global_view_path=None,
        prompt_mode="normal",
        return_answer=False,
        gate_reason=None,
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

            if self.use_dynamic_canvas:
                user_prompt = build_dynamic_anchor_aware_user_prompt(
                    query=caption,
                    parsed_query=json.dumps(parsed_query or {}, ensure_ascii=False, indent=2),
                    anchor_summary=anchor_summary or "",
                    candidate_summary=candidate_summary,
                    n_images=n_images,
                    has_final_global_view=final_global_view_path is not None,
                )

                if prompt_mode == "final_local_gate":
                    system_prompt = FINAL_LOCAL_GATE_PROMPT
                elif prompt_mode == "final_global_decision":
                    system_prompt = FINAL_GLOBAL_DECISION_PROMPT
                    if gate_reason:
                        user_prompt += (
                            "\n\nThe local-only gate requested the auxiliary global view for this reason:\n"
                            f"{gate_reason}\n"
                        )
                else:
                    system_prompt = DYNAMIC_ANCHOR_AWARE_SYSTEM_PROMPT
            else:
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
            
        content = [
            {"type": "text", "text": user_prompt},
        ]

        # Candidate dynamic canvases.
        for x in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{x[1]}",
                    "detail": "high",
                },
            })

        # Additional final-round global auxiliary view.
        # This is NOT a candidate image.
        if final_global_view_path is not None and os.path.exists(final_global_view_path):
            final_global_base64 = encode_image_to_base64(final_global_view_path)

            content.append({
                "type": "text",
                "text": (
                    "Additional auxiliary global view for the FINAL round only. "
                    "This image is NOT a candidate image. "
                    "Use it only to check global spatial layout. "
                    "The final answer must still be one of the valid candidate image_id values."
                ),
            })

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{final_global_base64}",
                    "detail": "high",
                },
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": content,
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
                    if return_answer:
                        return image_id, answer
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

        if return_answer:
            return -1, {}
        return -1
