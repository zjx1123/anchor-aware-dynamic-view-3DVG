# seqvlm/final_global_view_builder.py

import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw


VALID_VIEWS = ["top", "down", "up", "left", "right"]


class FinalGlobalViewBuilder:
    """
    Build one auxiliary global view for the final VLM round.

    This image is NOT a candidate image.
    It only provides global spatial context for the final-round candidates.
    """

    def __init__(
        self,
        rendered_root: str,
        out_root: str,
        max_anchors: int = 3,
    ):
        self.rendered_root = rendered_root
        self.out_root = out_root
        self.max_anchors = max_anchors
        os.makedirs(self.out_root, exist_ok=True)

    def _load_json(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _hash(self, query: str, candidate_ids: List[int], anchor_ids: List[int]) -> str:
        s = f"{query}__cands_{candidate_ids}__anchors_{anchor_ids}"
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

    def _get_view_info(
        self,
        rendered_view_meta: Dict[str, Any],
        proposal_id: int,
        view_name: str,
    ) -> Optional[Dict[str, Any]]:
        item = rendered_view_meta.get(str(proposal_id))
        if item is None:
            return None

        view_info = item.get("views", {}).get(view_name, {})
        if not view_info.get("visible", False):
            return None

        if "bbox_xyxy" not in view_info:
            return None

        return view_info

    def _relation_bias(self, parsed_query: Optional[Dict[str, Any]], view_name: str) -> float:
        """
        A light heuristic:
        - left/right/front/behind/north/south/east/west: BEV top view is often useful.
        - on/under/above/below: oblique views are often more useful.
        """
        if not parsed_query:
            return 0.0

        text = json.dumps(parsed_query, ensure_ascii=False).lower()

        horizontal_words = [
            "left", "right", "front", "behind",
            "north", "south", "east", "west",
            "near", "next to", "beside", "between",
        ]
        vertical_words = [
            "on", "under", "above", "below", "top",
        ]

        has_horizontal = any(w in text for w in horizontal_words)
        has_vertical = any(w in text for w in vertical_words)

        if has_horizontal and view_name == "top":
            return 2.0

        if has_vertical and view_name != "top":
            return 1.0

        return 0.0

    def _choose_best_view(
        self,
        rendered_view_meta: Dict[str, Any],
        final_candidates: List[Dict[str, int]],
        anchor_infos: List[Dict[str, Any]],
        parsed_query: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Score each global view by:
        - how many final target candidates are visible
        - how many anchors are visible
        - projected bbox area
        - light relation-type bias
        """
        best_view = None
        best_score = -1e9

        anchor_ids = [int(a["proposal_id"]) for a in anchor_infos[: self.max_anchors]]

        for view_name in VALID_VIEWS:
            target_visible = 0
            anchor_visible = 0
            area_sum = 0.0

            for c in final_candidates:
                proposal_id = int(c["proposal_id"])
                info = self._get_view_info(rendered_view_meta, proposal_id, view_name)
                if info is not None:
                    target_visible += 1
                    area_sum += float(info.get("area", 0.0))

            for aid in anchor_ids:
                info = self._get_view_info(rendered_view_meta, aid, view_name)
                if info is not None:
                    anchor_visible += 1
                    area_sum += 0.5 * float(info.get("area", 0.0))

            if target_visible == 0:
                continue

            score = (
                5.0 * target_visible
                + 2.0 * anchor_visible
                + 0.0005 * area_sum
                + self._relation_bias(parsed_query, view_name)
            )

            if score > best_score:
                best_score = score
                best_view = view_name

        return best_view

    def _draw_box(
        self,
        draw: ImageDraw.ImageDraw,
        bbox,
        label: str,
        color,
        image_w: int,
        image_h: int,
    ):
        x1, y1, x2, y2 = [int(v) for v in bbox]

        x1 = max(0, min(image_w - 1, x1))
        y1 = max(0, min(image_h - 1, y1))
        x2 = max(0, min(image_w - 1, x2))
        y2 = max(0, min(image_h - 1, y2))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        label_w = max(80, len(label) * 9 + 10)
        label_h = 24
        lx1 = x1
        ly1 = max(0, y1 - label_h)
        lx2 = min(image_w - 1, lx1 + label_w)
        ly2 = min(image_h - 1, ly1 + label_h)

        draw.rectangle([lx1, ly1, lx2, ly2], fill=(255, 255, 255))
        draw.text((lx1 + 5, ly1 + 5), label, fill=color)

    def build(
        self,
        scene_id: str,
        query: str,
        final_candidates: List[Dict[str, int]],
        anchor_infos: List[Dict[str, Any]],
        parsed_query: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        final_candidates example:
        [
            {"image_id": 3, "proposal_id": 27},
            {"image_id": 8, "proposal_id": 41},
        ]

        Important:
        image_id is the id VLM must output.
        proposal_id is the Mask3D proposal id used for drawing bbox.
        """
        if not final_candidates:
            return None

        scene_dir = os.path.join(self.rendered_root, scene_id)
        rendered_meta_path = os.path.join(scene_dir, "rendered_view_meta.json")
        rendered_view_meta = self._load_json(rendered_meta_path)

        if not rendered_view_meta:
            return None

        anchor_ids = [int(a["proposal_id"]) for a in anchor_infos[: self.max_anchors]]
        candidate_ids = [int(c["image_id"]) for c in final_candidates]

        out_dir = os.path.join(
            self.out_root,
            scene_id,
            self._hash(query, candidate_ids, anchor_ids),
        )
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, "final_global_view.jpg")
        if os.path.exists(out_path):
            return out_path

        best_view = self._choose_best_view(
            rendered_view_meta=rendered_view_meta,
            final_candidates=final_candidates,
            anchor_infos=anchor_infos,
            parsed_query=parsed_query,
        )

        if best_view is None:
            return None

        image_path = os.path.join(scene_dir, f"_mesh_{best_view}.png")
        if not os.path.exists(image_path):
            return None

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        image_w, image_h = img.size

        # Draw final-round target candidates in red.
        for c in final_candidates:
            image_id = int(c["image_id"])
            proposal_id = int(c["proposal_id"])

            info = self._get_view_info(rendered_view_meta, proposal_id, best_view)
            if info is None:
                continue

            # Label must contain selectable image_id.
            label = f"id={image_id}|pid={proposal_id}"
            self._draw_box(
                draw=draw,
                bbox=info["bbox_xyxy"],
                label=label,
                color=(255, 0, 0),
                image_w=image_w,
                image_h=image_h,
            )

        # Draw anchor/reference objects in blue.
        for a in anchor_infos[: self.max_anchors]:
            anchor_id = int(a["proposal_id"])

            info = self._get_view_info(rendered_view_meta, anchor_id, best_view)
            if info is None:
                continue

            label = f"A:{anchor_id}"
            self._draw_box(
                draw=draw,
                bbox=info["bbox_xyxy"],
                label=label,
                color=(0, 80, 255),
                image_w=image_w,
                image_h=image_h,
            )

        # Add title.
        title = f"Auxiliary global view: {best_view} | red=final candidates, blue=anchors"
        draw.rectangle([8, 8, min(image_w - 1, 720), 40], fill=(255, 255, 255))
        draw.text((16, 16), title, fill=(0, 0, 0))

        img.save(out_path)
        return out_path