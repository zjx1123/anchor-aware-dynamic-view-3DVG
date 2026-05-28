'''
【用途】

从 caption 中解析：

target category
anchor categories
spatial relations
target attributes
raw query

第一版用 LLM parser。为了避免每个样本重复 API 开销，加 JSON cache。
'''
import os
import json
import hashlib
from typing import Dict, Any, List

from api import invoke_api


QUERY_PARSE_SYSTEM_PROMPT = """
You are a 3D visual grounding query parser.
Given a natural language query, extract the target object and reference/anchor objects.

Return ONLY valid JSON. Do not use markdown.

Schema:
{
  "target": {
    "category": str,
    "attributes": [str]
  },
  "anchors": [
    {
      "category": str,
      "attributes": [str],
      "relation_to_target": str
    }
  ],
  "relations": [
    {
      "subject": str,
      "relation": str,
      "object": str
    }
  ],
  "view_constraints": [str]
}

Rules:
1. The target is the referred object to be localized.
2. Anchors are objects used to identify the target, such as table, door, window, bed, sofa, cabinet.
3. If no explicit anchor exists, return an empty anchors list.
4. Use simple object category names.
5. Do not invent anchors that are not implied by the query.
"""


QUERY_PARSE_USER_PROMPT = """
Query:
{query}

Return JSON:
"""


def _cache_key(query: str) -> str:
    return hashlib.md5(query.encode("utf-8")).hexdigest()


class QueryParser:
    def __init__(
        self,
        vlm_model: str,
        cache_dir: str = "../data/cache/query_parse_scanrefer",
        max_retry: int = 3,
    ):
        self.vlm_model = vlm_model
        self.cache_dir = cache_dir
        self.max_retry = max_retry
        os.makedirs(self.cache_dir, exist_ok=True)

    def parse(self, query: str, fallback_target: str = None) -> Dict[str, Any]:
        cache_path = os.path.join(self.cache_dir, _cache_key(query) + ".json")

        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        messages = [
            {"role": "system", "content": QUERY_PARSE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": QUERY_PARSE_USER_PROMPT.format(query=query),
            },
        ]

        parsed = None
        last_err = None

        for _ in range(self.max_retry):
            try:
                output = invoke_api(self.vlm_model, messages)
                parsed = json.loads(output["answer"])
                parsed = self._sanitize(parsed, query, fallback_target)
                break
            except Exception as e:
                last_err = e
                messages.append({
                    "role": "user",
                    "content": "Your previous answer was invalid. Return ONLY valid JSON.",
                })

        if parsed is None:
            parsed = self._fallback(query, fallback_target, str(last_err))

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)

        return parsed

    def _sanitize(
        self,
        parsed: Dict[str, Any],
        query: str,
        fallback_target: str = None,
    ) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            return self._fallback(query, fallback_target, "parsed is not dict")

        target = parsed.get("target", {})
        if not isinstance(target, dict):
            target = {}

        if not target.get("category"):
            target["category"] = fallback_target or ""

        if "attributes" not in target or not isinstance(target["attributes"], list):
            target["attributes"] = []

        anchors = parsed.get("anchors", [])
        if not isinstance(anchors, list):
            anchors = []

        clean_anchors = []
        for a in anchors:
            if not isinstance(a, dict):
                continue
            cat = a.get("category", "")
            if not cat:
                continue
            clean_anchors.append({
                "category": cat,
                "attributes": a.get("attributes", []) if isinstance(a.get("attributes", []), list) else [],
                "relation_to_target": a.get("relation_to_target", ""),
            })

        relations = parsed.get("relations", [])
        if not isinstance(relations, list):
            relations = []

        view_constraints = parsed.get("view_constraints", [])
        if not isinstance(view_constraints, list):
            view_constraints = []

        return {
            "raw_query": query,
            "target": target,
            "anchors": clean_anchors,
            "relations": relations,
            "view_constraints": view_constraints,
        }

    def _fallback(
        self,
        query: str,
        fallback_target: str = None,
        err: str = "",
    ) -> Dict[str, Any]:
        return {
            "raw_query": query,
            "target": {
                "category": fallback_target or "",
                "attributes": [],
            },
            "anchors": [],
            "relations": [],
            "view_constraints": [],
            "parse_error": err,
        }