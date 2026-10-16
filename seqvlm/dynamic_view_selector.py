'''
【用途】：读取离线 evidence pool, 根据 query + target proposal + anchor candidates + relation 动态生成 query-specific canvas。
【核心职责】：
1. 读取 crop_pool_meta_scanrefer/{scene_id}.json
2. 读取 view_meta_scanrefer/{scene_id}.json
3. 根据 target proposal 选 appearance crops
4. 根据 target-anchor 同屏帧生成 relation views
5. 可选加入 global/full-frame context view
6. 拼成 query-specific canvas
7. 返回 canvas_path 给 adaptive_predictor.py
'''

import os
import json
import cv2  
import hashlib
import shutil
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class DynamicViewSelector:
    """
    Build query-specific canvas for each target proposal.

    Input:
        - crop_images
        - crop_pool_meta_scanrefer
        - view_meta_scanrefer
        - posed_images_rgb_pose

    Output:
        - dynamic_canvas_root/{scene_id}/{query_hash}/{target_id}/canvas.jpg
    """

    def __init__(
        self,
        crop_image_root: str,
        crop_pool_meta_root: str,
        view_meta_root: str,
        posed_image_root: str,
        canvas_root: str,
        canvas_k: int = 5,
        num_appearance_views: int = 3,
        num_relation_views: int = 2,
        use_global_context: bool = True,
    ):
        self.crop_image_root = crop_image_root
        self.crop_pool_meta_root = crop_pool_meta_root
        self.view_meta_root = view_meta_root
        self.posed_image_root = posed_image_root
        self.canvas_root = canvas_root

        self.canvas_k = canvas_k
        self.num_appearance_views = num_appearance_views
        self.num_relation_views = num_relation_views
        self.use_global_context = use_global_context

        os.makedirs(self.canvas_root, exist_ok=True)

    def _load_json(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _query_hash(self, query: str, target_id: int) -> str:
        s = f"{query}__target_{target_id}"
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

    def _score_view(self, item: Dict[str, Any]) -> float:
        quality_bonus = 1.0 if item.get("quality") == "strong" else 0.3
        return (
            2.0 * quality_bonus
            + 1.0 * float(item.get("area_ratio", 0.0))
            + 0.002 * float(item.get("visible_points", 0.0))
        )


    def _draw_appearance_crop(
        self,
        scene_id: str,
        target_id: int,
        item: Dict[str, Any],
        out_path: str,
    ) -> Optional[str]:
        """
        Draw the target red box on an appearance crop.

        bbox_xyxy: target bbox in the original RGB frame
        crop_xyxy: crop region in the original RGB frame
        """
        src_path = os.path.join(
            self.crop_image_root,
            scene_id,
            str(target_id),
            item["file_name"],
        )

        if not os.path.exists(src_path):
            return None

        img = cv2.imread(src_path)
        if img is None:
            return None

        bbox = item.get("bbox_xyxy")
        crop_bbox = item.get("crop_xyxy")

        if bbox is None or crop_bbox is None:
            print(
                f"[AppearanceBox] missing bbox metadata: "
                f"scene={scene_id}, target={target_id}, "
                f"file={item.get('file_name')}"
            )
            return None

        # Coordinates in the original full-frame image
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop_x1, crop_y1, crop_x2, crop_y2 = [
            int(v) for v in crop_bbox
        ]

        # Convert full-frame bbox to crop-local coordinates
        local_x1 = x1 - crop_x1
        local_y1 = y1 - crop_y1
        local_x2 = x2 - crop_x1
        local_y2 = y2 - crop_y1

        h, w = img.shape[:2]

        # Clip the box to the crop boundary
        local_x1 = max(0, min(local_x1, w - 1))
        local_y1 = max(0, min(local_y1, h - 1))
        local_x2 = max(0, min(local_x2, w - 1))
        local_y2 = max(0, min(local_y2, h - 1))

        if local_x2 <= local_x1 or local_y2 <= local_y1:
            print(
                f"[AppearanceBox] invalid local bbox: "
                f"scene={scene_id}, target={target_id}, "
                f"bbox={[local_x1, local_y1, local_x2, local_y2]}"
            )
            return None

        # Target: red in BGR
        cv2.rectangle(
            img,
            (local_x1, local_y1),
            (local_x2, local_y2),
            (0, 0, 255),
            3,
        )

        # Keep the label inside the crop
        if local_y1 >= 24:
            text_y = local_y1 - 7
        else:
            text_y = min(local_y1 + 22, h - 5)

        cv2.putText(
            img,
            f"T:{target_id}",
            (local_x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img)

        return out_path
    
    def _select_target_appearance_views(
        self,
        scene_id: str,
        target_id: int,
        crop_pool_meta: Dict[str, Any],
        out_dir: str,
    ) -> Tuple[List[str], set]:
        """
        Select target appearance crops and draw a red target box on each crop.
        """
        obj_key = str(target_id)
        items = crop_pool_meta.get(obj_key, [])

        if not items:
            return [], set()

        items = sorted(
            items,
            key=self._score_view,
            reverse=True,
        )

        image_paths = []
        used_file_names = set()

        for item in items:
            if len(image_paths) >= self.num_appearance_views:
                break

            file_name = item["file_name"]
            image_name = str(item.get("image_name", len(image_paths)))

            out_path = os.path.join(
                out_dir,
                f"appearance_{len(image_paths):02d}_{image_name}.jpg",
            )

            marked_path = self._draw_appearance_crop(
                scene_id=scene_id,
                target_id=target_id,
                item=item,
                out_path=out_path,
            )

            if marked_path is not None:
                image_paths.append(marked_path)
                used_file_names.add(file_name)

        return image_paths, used_file_names
    
    def _index_views_by_frame(self, views: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return {str(v["image_name"]): v for v in views if "image_name" in v}

    def _find_best_anchor(self, anchor_infos: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        First version:
        choose the highest-score anchor proposal from parsed anchor candidates.
        Later can be extended by relation type or category priority.
        """
        if not anchor_infos:
            return None

        anchor_infos = sorted(
            anchor_infos,
            key=lambda x: float(x.get("score", 0.0)),
            reverse=True,
        )
        return anchor_infos[0]

    def _select_relation_frames(
        self,
        target_id: int,
        anchor_id: int,
        view_meta: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any], Dict[str, Any], float]]:
        """
        Select frames where target and anchor are co-visible.
        """
        target_views = view_meta.get(str(target_id), [])
        anchor_views = view_meta.get(str(anchor_id), [])

        if not target_views or not anchor_views:
            return []

        target_by_frame = self._index_views_by_frame(target_views)
        anchor_by_frame = self._index_views_by_frame(anchor_views)

        common_frames = sorted(set(target_by_frame.keys()) & set(anchor_by_frame.keys()))
        scored = []

        for frame in common_frames:
            tv = target_by_frame[frame]
            av = anchor_by_frame[frame]

            target_score = self._score_view(tv)
            anchor_score = self._score_view(av)

            # co-visible score
            score = target_score + 0.8 * anchor_score + 2.0
            scored.append((frame, tv, av, score))

        scored = sorted(scored, key=lambda x: x[-1], reverse=True)
        return scored[: self.num_relation_views]

    def _draw_relation_view(
        self,
        scene_id: str,
        frame_name: str,
        target_id: int,
        anchor_id: int,
        target_bbox: List[int],
        anchor_bbox: List[int],
        out_path: str,
    ) -> Optional[str]:
        """
        Draw target red box and anchor blue box on the original full-frame RGB.
        """
        img_path = os.path.join(self.posed_image_root, scene_id, f"{frame_name}.jpg")
        if not os.path.exists(img_path):
            return None

        img = cv2.imread(img_path)
        if img is None:
            return None

        # target: red
        x1, y1, x2, y2 = [int(x) for x in target_bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            img,
            f"T:{target_id}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        # anchor: blue
        ax1, ay1, ax2, ay2 = [int(x) for x in anchor_bbox]
        cv2.rectangle(img, (ax1, ay1), (ax2, ay2), (255, 0, 0), 3)
        cv2.putText(
            img,
            f"A:{anchor_id}",
            (ax1, max(0, ay1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img)
        return out_path

    def _resize_keep_ratio(self, img, target_w: int = 512):
        h, w = img.shape[:2]
        if w == target_w:
            return img
        scale = target_w / max(w, 1)
        new_h = max(1, int(h * scale))
        return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)

    def _stitch_vertical(self, image_paths: List[str], out_path: str) -> Optional[str]:
        imgs = []
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            img = self._resize_keep_ratio(img, target_w=512)
            imgs.append(img)

        if not imgs:
            return None

        canvas = cv2.vconcat(imgs)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, canvas)
        return out_path

    def build_query_specific_canvas(
        self,
        scene_id: str,
        target_proposal_id: int,
        query: str,
        parsed_query: Optional[Dict[str, Any]],
        anchor_infos: Optional[List[Dict[str, Any]]],
        candidate_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Main API called by adaptive_predictor.py.

        Return:
            canvas_path or None
        """
        parsed_query = parsed_query or {}
        anchor_infos = anchor_infos or []

        q_hash = self._query_hash(query, target_proposal_id)
        out_dir = os.path.join(
            self.canvas_root,
            scene_id,
            q_hash,
            str(target_proposal_id),
        )
        canvas_path = os.path.join(out_dir, "canvas.jpg")

        # cache
        if os.path.exists(canvas_path):
            return canvas_path

        crop_pool_path = os.path.join(self.crop_pool_meta_root, f"{scene_id}.json")
        view_meta_path = os.path.join(self.view_meta_root, f"{scene_id}.json")

        crop_pool_meta = self._load_json(crop_pool_path)
        view_meta = self._load_json(view_meta_path)

        if not crop_pool_meta or not view_meta:
            return None

        selected_images = []

        # 1. target appearance crops
        appearance_imgs, used_appearance_files = (
            self._select_target_appearance_views(
                scene_id=scene_id,
                target_id=target_proposal_id,
                crop_pool_meta=crop_pool_meta,
                out_dir=out_dir,
            )
        )
        selected_images.extend(appearance_imgs)

        # 2. target-anchor relation views
        best_anchor = self._find_best_anchor(anchor_infos)
        relation_imgs = []

        if best_anchor is not None:
            anchor_id = int(best_anchor["proposal_id"])
            relation_frames = self._select_relation_frames(
                target_id=target_proposal_id,
                anchor_id=anchor_id,
                view_meta=view_meta,
            )

            for rank, (frame, tv, av, score) in enumerate(relation_frames):
                rel_path = os.path.join(out_dir, f"relation_{rank:02d}_{frame}.jpg")
                p = self._draw_relation_view(
                    scene_id=scene_id,
                    frame_name=frame,
                    target_id=target_proposal_id,
                    anchor_id=anchor_id,
                    target_bbox=tv["bbox_xyxy"],
                    anchor_bbox=av["bbox_xyxy"],
                    out_path=rel_path,
                )
                if p is not None:
                    relation_imgs.append(p)

        selected_images.extend(relation_imgs)

        # 3. Fallback: fill empty slots with additional boxed appearance crops
        if len(selected_images) < self.canvas_k:
            all_target_items = crop_pool_meta.get(
                str(target_proposal_id),
                [],
            )
            all_target_items = sorted(
                all_target_items,
                key=self._score_view,
                reverse=True,
            )

            fill_rank = 0

            for item in all_target_items:
                if len(selected_images) >= self.canvas_k:
                    break

                file_name = item["file_name"]

                # Avoid repeating an appearance crop already selected above
                if file_name in used_appearance_files:
                    continue

                image_name = str(
                    item.get("image_name", fill_rank)
                )

                out_path = os.path.join(
                    out_dir,
                    f"appearance_fill_{fill_rank:02d}_{image_name}.jpg",
                )

                marked_path = self._draw_appearance_crop(
                    scene_id=scene_id,
                    target_id=target_proposal_id,
                    item=item,
                    out_path=out_path,
                )

                if marked_path is not None:
                    selected_images.append(marked_path)
                    used_appearance_files.add(file_name)
                    fill_rank += 1

        selected_images = selected_images[: self.canvas_k]

        if not selected_images:
            return None

        return self._stitch_vertical(selected_images, canvas_path)
