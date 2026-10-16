'''
VLM调用：Prompt
SeqVLM 的 VLM 决策是从一个 batch 的候选 canvas 中选择一个 image_id
VLM 输出格式固定为：
{
  "process": "...",
  "image_id": 1
}
'''
SYSTEM_PROMPT = """Imagine you are in a room and you are aksed to find one object.

Given a series of images and a query describing a specific object in the room, you need to analyze the images, and find an image that best fits the query.

Please note that each image is composed of sub-images displaying the object from multiple perspectives. In each sub-image, there is a red rectangle box highlighting the object, but the box may also contain other irrelevant objects. You need to make a selection by combining the object in the red rectangle box with surrounding environment from different perspective images.

Return the index of the image where the object is found, and describe the process of selecting this image.

Your response should be in the following format, and it should not include code block markers such as ```json.

{
  "process": "Explain the process of how you identified the room's features and located the target object",
  "image_id": 1 # Replace with the actual index based on the input order of images, starting from 0. 
}

Here is an example for you.

```
Input: 
Query: Find the black table that is surrounded by four chairs.
Here are the images of 3 possible objects.
[image_0, image_1, image_2]

Output:
{
  "process": "After carefully examining all the input images, I found only the tables in image_1, image_2 are black, but only the tables in image_2 has is surrounded by four chairs. So the correct object is the table in image_2",
  "image_id": 2
}

```

Here are some tips:
# Please follow the format of the example strictly
# If there is no object that fully matches the query, select the most suitable one. 
# If the types of all objects are inconsistent with the query, output -1 in the value of image_id.

"""

USER_PROMPT = """Query: {query}
Here are the images of {n_images} possible objects."""

IMAGE_ID_INVALID_PROMPT = """The image_id {image_id} you selected does not exist. Did you perhaps see it incorrectly? Please reconsider and select another image. Remember to reply using JSON format with the two keys "process", "image_id" as required before."""

WRONG_FORMAT_PROMPT = """The answer contains extra characters. Please follow the format of the example strictly."""

ANCHOR_AWARE_SYSTEM_PROMPT = """
Imagine you are in a 3D indoor scene and you are asked to find one target object.
You are given:
1. A natural language query.
2. Several candidate target object images. Each image is a stitched multi-view canvas of one candidate object.
3. Parsed reference/anchor objects from the same 3D scene, including their object ids, classes, 3D locations, and relations to the target.

In each candidate image, the red rectangle highlights the candidate target object.
The reference/anchor objects may or may not be visible in the candidate images.
You must use BOTH:
- visual evidence from the candidate images,
- spatial/reference-object information from the anchor summary.

Important:
- If the query contains reference objects such as table, door, window, bed, sofa, cabinet, wall, monitor, etc., you must explicitly check whether the candidate target satisfies the relation with those anchors.
- In scenes with multiple similar target objects, do not select only by appearance. Use anchors and 3D spatial cues to disambiguate.
- Select the candidate image_id that best matches the full query.

Return ONLY valid JSON, no markdown, no code block.
The JSON format must be:
{
  "process": "Explain how you checked target appearance, anchors, and relations.",
  "image_id": 0
}
"""

# dynamic anchor-aware prompt
DYNAMIC_ANCHOR_AWARE_SYSTEM_PROMPT = """
Imagine you are in a 3D indoor scene and you are asked to find one target object.

You are given several candidate images. Each candidate image is a query-specific canvas
constructed for one candidate target proposal.

Important visual marks:
- RED boxes indicate the candidate target object.
- BLUE boxes indicate selected reference/anchor objects mentioned in the query.
- The BLUE box may show only ONE of the reference objects from the query, not necessarily all of them.
- Some sub-images focus on target appearance.
- Some sub-images show target-anchor spatial relations in the original full-frame view.

You must use BOTH:
1. local target appearance, such as category, color, texture, material, shape;
2. target-anchor relation evidence, such as near, under, on, left of, right of, behind, in front of;
3. global scene context if visible.

Do not select only by target appearance when the query contains reference objects.
First verify that the RED-boxed object matches the target category and attributes.
Use a spatial relation only when the target and the correct anchor are visibly grounded in the same relation view.
Missing or unclear relation evidence is unknown, not a contradiction; never infer linguistic directions from camera orientation.
You must check whether the red-box target satisfies the full query. Do not focus only on the blue-box anchor; the blue box is only visual evidence for one selected reference object. 
Use only visible evidence and the provided text. Do not infer relations from unavailable geometry. 
Return ONLY valid JSON, no markdown, no code block.
The JSON format must be:
{ "process": "Explain how you checked target appearance, anchors, and spatial relations.", "image_id": 0 }
"""

def build_anchor_aware_user_prompt(
    query: str,
    parsed_query: str,
    anchor_summary: str,
    candidate_summary: str,
    n_images: int,
) -> str:
    json_example = (
        '{\n'
        '  "process": "...",\n'
        '  "image_id": 0\n'
        '}'
    )
    return (
        f"Query:\n{query}\n"
        f"Parsed query:\n{parsed_query}\n"
        f"Current candidate batch:\n{candidate_summary}\n"
        f"There are {n_images} candidate images in this batch.\n"
        "The image_id values refer to the original candidate indices shown in the "
        "candidate summary, not necessarily 0..n-1.\n"
        "Please select the best image_id.\n"
        f"Return ONLY valid JSON:\n{json_example}\n"
    )


def build_dynamic_anchor_aware_user_prompt(
    query: str,
    parsed_query: str,
    anchor_summary: str,
    candidate_summary: str,
    n_images: int,
    has_final_global_view: bool = False,
) -> str:
    json_example = (
        '{\n'
        '  "process": "...",\n'
        '  "image_id": 0\n'
        '}'
    )

    final_global_note = ""
    if has_final_global_view:
        final_global_note = (
            "\nFinal-round auxiliary global view:\n"
            "- After the candidate dynamic canvases, you are given ONE additional global view.\n"
            "- This global view is NOT a candidate image.\n"
            "RED boxes mark candidate target objects.\n"
            "BLUE boxes mark selected reference/anchor hypotheses when visible.\n"
            "The BLUE-boxed object is an anchor hypothesis, not a verified anchor.\n"
            "Its attributes and relation to the target must be checked visually.\n"
            "If a required anchor is not visibly grounded, treat that relation as unverified.\n"
            "Do not infer missing anchor relations.\n"
            "Select the candidate that best satisfies the full query using directly visible evidence.\n"
            "- Use this global view only to verify global spatial layout and target-anchor relations.\n"
            "- Make the final decision mainly based on the candidate dynamic canvases.\n"
            "- Your output image_id must be one of the candidate image_id values in the current candidate batch.\n"
        )

    return (
        f"Query:\n{query}\n\n"
        f"Parsed query:\n{parsed_query}\n"
        f"Anchor/reference object summary:\n{anchor_summary}\n"
        f"Current candidate batch:\n{candidate_summary}\n"
        f"There are {n_images} candidate images in this batch.\n"
        "Each candidate image is a dynamic canvas for one target proposal.\n"
        "RED boxes mark candidate target objects.\n"
        "BLUE boxes mark selected reference/anchor objects when visible.\n"
        "The BLUE box may show only one reference object from the query, not all reference objects.\n"
        "If a reference object is not visibly grounded, treat its relation as unverified rather than inferring it.\n"
        "Do not put all attention only on the blue-box anchor; select the object that best satisfies the full text query.\n"
        "The image_id values refer to the original candidate indices shown in the candidate summary, "
        "not necessarily 0..n-1.\n"
        f"{final_global_note}\n"
        "Please select the best image_id.\n"
        f"Return ONLY valid JSON:\n{json_example}\n"
    )

FINAL_LOCAL_GATE_PROMPT = """
You are in the final decision stage of a 3D visual grounding task.

You are given several candidate dynamic canvases. Each candidate image contains a red box marking the target candidate, and may contain blue boxes marking local anchor objects.

Your task has two parts:

1. Select the best candidate image_id using the local dynamic canvases.
2. Decide whether an auxiliary global view is necessary for a reliable final decision.

Use the local dynamic canvases as the primary evidence. First judge target category, appearance, local context, visible anchors, and non-directional spatial relations.

Set "need_global" to true when the local canvases are insufficient and the query requires room-level or scene-level context.

Set "need_global" to false when the decision can be made from local evidence, or when global view is likely to be unreliable or misleading.

Return only valid JSON in this format:
{
  "process": "...",
  "need_global": true or false,
  "global_need_reason": "...",
  "image_id": 0
}
"""

FINAL_GLOBAL_DECISION_PROMPT = """
You are in the final decision stage of a 3D visual grounding task.

You are given:
1. Candidate dynamic canvases, where red boxes mark the target candidates.
2. An auxiliary global view, where red boxes mark final candidate objects and blue boxes mark non-candidate anchor objects.

Use the candidate dynamic canvases as the primary evidence. The global view is only auxiliary context.

Use the global view only to verify the exact missing fact stated by the gate.

Do not use the global view to infer viewpoint-dependent directions such as left/right/front/back/east/west/north/south. Do not use the global view to judge facing direction or object orientation. For these cases, rely on local dynamic canvases and candidate summaries.

If the target and anchor have the same category, do not swap the target with the anchor. Red-boxed final candidates are possible targets. Blue boxes are only auxiliary anchors. The global view should not override clear local target evidence.
Check only the missing fact stated by the gate.
Change the local selection only if the global view clearly proves that another candidate satisfies that fact and the local candidate does not.
If the global evidence is ambiguous, keep the local selection.

If the global view and local dynamic canvases conflict, prefer the local dynamic canvases unless the global view provides decisive non-directional layout evidence.

Return only valid JSON in this format:
{
  "process": "...",
  "image_id": 0
}
"""