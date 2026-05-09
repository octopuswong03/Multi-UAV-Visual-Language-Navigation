import re
import sys
import json
sys.path.append("../../")
from airsim_plugin.airsim_settings import DefaultAirsimActionNames, DefaultAirsimActionCodes, ObservationDirections

action_space = "Action Space:\nforward (go straight), left (rotate left), right (rotate right), stop (end navigation)\n\n"
prompt_template = 'Navigation Instructions:\n"{}"\nAction Sequence:\n'


def parse_viewpoint_response(resp):
    cleaned_resp = resp.strip('`json\n').strip('`')
    json_resp = json.loads(cleaned_resp)
    reformat_resp = {}
    for k in json_resp:
        if "slightly" in k and "left" in k:
            reformat_resp["slightly_left"] = json_resp[k]
        elif "slightly" in k and "right" in k:
            reformat_resp["slightly_right"] = json_resp[k]
        elif "left" in k:
            reformat_resp["left"] = json_resp[k]
        elif "right" in k:
            reformat_resp["right"] = json_resp[k]
        elif "front" in k:
            reformat_resp["front"] = json_resp[k]

    return reformat_resp


def parse_viewpoint_response_v2(resp):
    cleaned_resp = resp.strip('`json\n').strip('`')
    cleaned_resp = cleaned_resp.replace("True", "true").replace("False", "false")

    json_resp = json.loads(cleaned_resp)
    reformat_resp = {}
    if "is_found" not in json_resp:
        print("Warning: LLM output missing 'is_found'")
        reformat_resp["is_found"] = False
    else:
        reformat_resp["is_found"] = json_resp["is_found"]

    for k in json_resp:
        if "slightly" in k and "left" in k:
            reformat_resp["slightly_left"] = json_resp[k]
        elif "slightly" in k and "right" in k:
            reformat_resp["slightly_right"] = json_resp[k]
        elif "left" in k:
            reformat_resp["left"] = json_resp[k]
        elif "right" in k:
            reformat_resp["right"] = json_resp[k]
        elif "front" in k:
            reformat_resp["front"] = json_resp[k]

    return reformat_resp


def visual_observation_prompt_builder():
    # landmark_str = ""
    # for landmark in landmarks:
    #     landmark_str += "{}: {}\n".format(landmark, landmarks[landmark])

    prompt_str = (
            "Given an image, please describe the object and scenes with its characteristics. \n"
            # + "The characteristics must include its color, texture, height and width. \n"
            # + "Besides, you need to tell whether you observed landmarks provided below: \n"
            # + "{landmark_str}".format(landmark_str=landmark_str)
            + "\nYour output should be in following format: \n"
            + "[{observed object}: {characteristics};[{observed object}: {characteristics}...]\n"
            # + '{"observed landmark": [{landmark1}, {landmark2}, {landmark3}...]}'
            + """
        Here are some output examples: 

        [a skycraper: silver, smooth, tall and slim; cloudy sky: white, fulfilled, high and wide; garden: green and red, messy, short and small]
        """
    )

    return prompt_str


def cot_prompt_builder_p1(
        navigation_instruction,
        prev_actions,
):
    prompt_str = f"""
    You are an AI agent helping to control the drone to finish the navigation task.
    Your navigation task is {navigation_instruction}.
    Your previous action sequence is {prev_actions}.
    Based on your navigation task and previous action, what's your current sub-goal to reach.
    """
    return prompt_str


def cot_prompt_builder_p2(
        navigation_instruction,
        prev_actions,
        current_subgoal,
        current_observation,
        current_position,
):
    prompt_str = f"""
    You are an AI agent helping to control the drone to finish the navigation task.
    Your current sub-goal to reach is {current_subgoal}.
    Your previous action sequence is {prev_actions}.
    Your current observation is {current_observation}.
    Your current height is {-int(current_position.z_val)} meters.
    Based on your current sub-goal and your current observation, whether you achieve the sub-goal?
    Your should answer 'YES' or 'NO', and provide the reason why you give such an answer.
    
    Example output: 
    YES: the drone already reached a high altitude.
    NO: the landmark in the sub-goal is not observed.
    """
    return prompt_str

def cot_prompt_builder_p3(
        navigation_instruction,
        prev_actions,
        current_subgoal,
        subgoal_status,
        current_observation,
):
    if subgoal_status is True:
        prompt_str = f"""
        You are an AI agent helping to control the drone to finish the navigation task.
        Your navigation task is {navigation_instruction}.
        So far, you've reach the sub-goal: {current_subgoal}.
        Based on your reached sub-goal and your navigation task, what's your next sub-goal to reach?
        
        
        """
    else:
        action_list = [act for act in DefaultAirsimActionCodes]
        prompt_str = f"""
        You are an AI agent helping to control the drone to finish the navigation task.
        Your navigation task is {navigation_instruction}.
        You need to reach the sub-goal: {current_subgoal}.
        Your current observation is: {current_observation};
        your previous action sequence is: {prev_actions}
        Based on your sub-goal, current observation and previous actions, 
        please provide the next action you should take. You can ONLY use ONE of the following actions: {action_list}. 
        Also, you need to provide the reason why you take the action.
        
        Example output: 
        MOVE_FORWARD: the building in front appeared for a lot of times with the target, I think it's helpful for finding the target.
        MOVE_LEFT: the target building is on the left
        MOVE_FORWARD: this place is arrived in earlier experience, so you should follow the earlier path.
        STOP: you have arrived at the target.
        GO_UP: follow the navigation instruction, take off
        GO_DOWN: the goal is reached, land to the floor.
        """

    return prompt_str



def open_ended_action_manager_prompt_builder_v2(
        navigation_instruction,
        current_text,
        prev_action=None,
        experience=None,
        relate_knowledge=None,
        fire=False,
        exclude_actions=None,
):
    action_list = [act for act in DefaultAirsimActionCodes]
    prompt_str = f"""
    You are an AI agent helping to control the drone to finish the navigation task.
    Your navigation task is {navigation_instruction}.
    Your current observation is {current_text}.
    Your previous action sequence is {prev_action}
    Based on your navigation task, current observation and previous action, please provide the next action you should take. You can ONLY use ONE of the following actions: {action_list}. Also, you need to provide the reason why you take the action.
    
    Example output: 
    MOVE_FORWARD: the building in front appeared for a lot of times with the target, I think it's helpful for finding the target.
    MOVE_LEFT: the target building is on the left
    MOVE_FORWARD: this place is arrived in earlier experience, so you should follow the earlier path.
    STOP: you have arrived at the target.
    GO_UP: follow the navigation instruction, take off
    GO_DOWN: the goal is reached, land to the floor.

    """

    return prompt_str


def subtask_action_manager_prompt_builder(subtask, finished_checkpoint, ongoing_checkpoint, pano_observation):
    action_list = [act for act in DefaultAirsimActionCodes]
    action_list = action_list[:-2]

    prompt_str = f"""
    You are an AI agent helping to control the drone to finish the navigation task.
    Your navigation task is {subtask}.
    The checkpoints you have finished are: {finished_checkpoint}.
    The ongoing checkpoint is: {ongoing_checkpoint}.
    Your current observation is {pano_observation}.
    
    Based on your navigation task progress and current observation, please provide the next action you should take.
    You can ONLY use ONE of the following actions: {action_list}. Also, you need to provide the reason why you take the action.
    
    Example output: 
    MOVE_FORWARD: the building in front appeared for a lot of times with the target, I think it's helpful for finding the target.
    MOVE_LEFT: the target building is on the left
    MOVE_FORWARD: this place is arrived in earlier experience, so you should follow the earlier path.
    STOP: you have arrived at the target.
    GO_UP: follow the navigation instruction, take off
    GO_DOWN: the goal is reached, land to the floor.
    """

    return prompt_str

def summarize_view_prompt_builder(full_view):
    assert len(full_view) == len(ObservationDirections)
    view_str = ""
    for i in range(len(full_view)):
        view_str += f"{ObservationDirections[i]}: {full_view[i]}\n"
    prompt_str = f"""
    You are an AI agent helping to control the drone to navigate the street view. 
    Your task is to summarize the view of the environment. You are told the view of the environment for 8 directions. Please summarize the following view into 1 paragraph, to reveal the overview of the current place.
    {view_str}
    """
    return prompt_str


def summarize_view_observation(full_view, collision_risk=[]):
    collision_risk = ["no" for _ in range(len(full_view))]

    assert len(full_view) == len(ObservationDirections)
    view_str = ""
    for i in range(len(full_view)):
        view_str += f"{ObservationDirections[i]}: {full_view[i]}, collision risk: {collision_risk[i]}\n"

    return view_str


def relative_spatial_prompt_builder(path):
    if len(path) == 0:
        return ""

    curr_loc = path[0]
    prev_loc = path[1]
    distance = path[2]
    rel_direction = path[3]
    prompt_str = f"{curr_loc} is located {distance} meters {rel_direction} of the {prev_loc} "

    return prompt_str


def landmark_memory_prompt_builder(instruction, landmarks):
    landmark_str = "\n".join(landmarks)
    landmark_path = []
    landmark_path_strs = []

    cnt = 1
    for i in range(len(landmarks)-1):
        # for j in range(i+1, len(landmarks)):
        #     landmark_path.append(f"{cnt}. <{landmarks[i]}> to <{landmarks[j]}>")
        #     cnt += 1
        landmark_path_strs.append(f"{cnt}. <{landmarks[i]}> to <{landmarks[i+1]}>")
        landmark_path.append([landmarks[i], landmarks[i+1]])
        cnt += 1

    landmark_path_prompt = "\n".join(landmark_path_strs)

    prompt = f"""
    Navigation instruction: {instruction}
    
    Landmarks in the instruction: 
    {landmark_str}
    
    Based on the instruction, describe the path between following path:
    {landmark_path_prompt}
    
    Your output format:
    1. <path>
    2. <path>
    ...
    """

    return prompt, landmark_path


def landmark_caption_prompt_builder(scene_objects):
    if len(scene_objects) == 0:
        prompt = f"""
List the objects that appears in the images. Each object use no more than 5 words to describe.

Example output:
object1.object2.object3
"""
    else:
        prompt = f"""
List the objects that appears in the image from the list below:
{scene_objects}.

Example output:
object1.object2.object3

Your output:
    """

    return prompt


def route_planning_prompt_builder(navigation_instruction, total_subgoals, traversed_subgoals, next_subgoal):
    total_subgoals_str = ", ".join(total_subgoals)
    traversed_subgoals_str = ", ".join(traversed_subgoals)

    prompt = f"""
You are a drone and your task is navigating to the described target location!

Navigation instruction: {navigation_instruction}
You need to sequentially traverse all the following subgoals to reach the target: {total_subgoals_str}
The subgoals you have traversed are: {traversed_subgoals_str}
Your next navigation subgoal: {next_subgoal}

Your visual observation from different viewpoints are provide above.

Based on the instruction, next navigation subgoal and observation, you need to plan your next waypoint. 
There are two situations: 
    If you find the next subgoal, you should output the viewpoint that observes the subgoal in JSON format. E.g.:
{{
    "is_found": true,
    "slightly left": subgoal
}}
    If you don't find the next subgoal, you should select 3 objects you will probably go next from your observations in descending order of probability to find the subgoal. E.g.:
{{
    "is_found": false,
    "front": "object 1",
    "left": "object 2",
    "slightly left": "object 3"
}}
    """

    return prompt


_ROUTE_VIEWPOINTS = ("left", "slightly_left", "front", "slightly_right", "right")


def _normalize_route_viewpoint_key(key):
    if not isinstance(key, str):
        return None
    k = key.strip().lower().replace("-", " ").replace("_", " ")
    k = " ".join(k.split())
    if "slightly" in k and "left" in k:
        return "slightly_left"
    if "slightly" in k and "right" in k:
        return "slightly_right"
    if k == "left" or (" left" in f" {k} " and "slightly" not in k):
        return "left"
    if k == "right" or (" right" in f" {k} " and "slightly" not in k):
        return "right"
    if "front" in k:
        return "front"
    return None


def _json_like_dump(data):
    if data is None:
        return "null"
    if isinstance(data, str):
        return data
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def route_planning_multievidence_prompt_builder(
        navigation_instruction,
        total_subgoals,
        traversed_subgoals,
        next_subgoal,
        semantic_context=None,
        spatial_context=None,
        temporal_context=None,
        collaborative_context=None,
):
    total_subgoals_str = ", ".join(total_subgoals)
    traversed_subgoals_str = ", ".join(traversed_subgoals)

    semantic_block = _json_like_dump(semantic_context)
    spatial_block = _json_like_dump(spatial_context)
    temporal_block = _json_like_dump(temporal_context)
    collaborative_block = _json_like_dump(collaborative_context)

    prompt = f"""
You are a multimodal navigation reasoner for a drone. Keep zero-shot reasoning and do not assume any extra training.

Task:
- Navigation instruction: {navigation_instruction}
- Ordered landmarks: {total_subgoals_str}
- Traversed landmarks: {traversed_subgoals_str}
- Current next landmark: {next_subgoal}

You must align FOUR evidence sources before deciding:
1) Semantic evidence (instruction and traversed landmarks)
2) Spatial evidence (multi-view observation, viewpoint geometry, current pose, landmark candidates)
3) Temporal evidence (recent flight trajectory and confirmed landmark history)
4) Multi-UAV evidence (other UAV views/hypotheses and cross-UAV hints; may be null in single-UAV mode)

Structured context (JSON-like):
[Semantic]
{semantic_block}

[Spatial]
{spatial_block}

[Temporal]
{temporal_block}

[Multi-UAV]
{collaborative_block}

Decision policy:
- Never rely on only one visual-language match.
- First check whether <{next_subgoal}> is already visible.
- If visible, set is_found=true and return the best matching viewpoint/object.
- If not visible, set is_found=false and propose likely proxy objects by viewpoint.
- Output a calibrated probability distribution over viewpoints for where the NEXT landmark is most likely to appear.
- Prefer consistency between semantic, spatial, temporal, and multi-UAV evidence.

Output JSON only (no markdown, no explanation):
{{
  "is_found": true/false,
  "left": "object or unknown",
  "slightly_left": "object or unknown",
  "front": "object or unknown",
  "slightly_right": "object or unknown",
  "right": "object or unknown",
  "next_landmark_prob": {{
    "left": float_0_to_1,
    "slightly_left": float_0_to_1,
    "front": float_0_to_1,
    "slightly_right": float_0_to_1,
    "right": float_0_to_1
  }},
  "evidence_alignment": {{
    "left": {{"semantic": float_0_to_1, "spatial": float_0_to_1, "temporal": float_0_to_1, "multi_uav": float_0_to_1, "overall": float_0_to_1, "object": "..." }},
    "slightly_left": {{"semantic": float_0_to_1, "spatial": float_0_to_1, "temporal": float_0_to_1, "multi_uav": float_0_to_1, "overall": float_0_to_1, "object": "..." }},
    "front": {{"semantic": float_0_to_1, "spatial": float_0_to_1, "temporal": float_0_to_1, "multi_uav": float_0_to_1, "overall": float_0_to_1, "object": "..." }},
    "slightly_right": {{"semantic": float_0_to_1, "spatial": float_0_to_1, "temporal": float_0_to_1, "multi_uav": float_0_to_1, "overall": float_0_to_1, "object": "..." }},
    "right": {{"semantic": float_0_to_1, "spatial": float_0_to_1, "temporal": float_0_to_1, "multi_uav": float_0_to_1, "overall": float_0_to_1, "object": "..." }}
  }}
}}

Rules:
- Probabilities in next_landmark_prob should sum to 1.0 (or very close due to rounding).
- Use "unknown" for viewpoints with no clear proxy object.
- Keep scores conservative when evidence conflicts.
"""
    return prompt


def parse_viewpoint_response_multievidence(resp):
    cleaned_resp = resp.strip('`json\n').strip('`')
    cleaned_resp = cleaned_resp.replace("True", "true").replace("False", "false")
    data = json.loads(cleaned_resp)

    is_found_raw = data.get("is_found", False)
    if isinstance(is_found_raw, str):
        is_found = is_found_raw.strip().lower() in ("1", "true", "yes")
    else:
        is_found = bool(is_found_raw)

    out = {
        "is_found": is_found,
        "viewpoint_objects": {vp: "unknown" for vp in _ROUTE_VIEWPOINTS},
        "viewpoint_scores": {vp: 0.0 for vp in _ROUTE_VIEWPOINTS},
        "next_landmark_prob": {vp: 0.0 for vp in _ROUTE_VIEWPOINTS},
        "evidence_alignment": {},
    }

    for k, v in data.items():
        vp = _normalize_route_viewpoint_key(k)
        if vp is None:
            continue
        out["viewpoint_objects"][vp] = str(v) if v is not None else "unknown"

    prob_block = data.get("next_landmark_prob", {})
    if isinstance(prob_block, dict):
        for k, v in prob_block.items():
            vp = _normalize_route_viewpoint_key(k)
            if vp is None:
                continue
            try:
                out["next_landmark_prob"][vp] = max(0.0, float(v))
            except Exception:
                pass

    align_block = data.get("evidence_alignment", {})
    if isinstance(align_block, dict):
        for k, v in align_block.items():
            vp = _normalize_route_viewpoint_key(k)
            if vp is None or not isinstance(v, dict):
                continue
            out["evidence_alignment"][vp] = v
            if "object" in v and isinstance(v["object"], str) and v["object"].strip():
                out["viewpoint_objects"][vp] = v["object"].strip()
            try:
                out["viewpoint_scores"][vp] = max(0.0, float(v.get("overall", 0.0)))
            except Exception:
                pass

    for vp in _ROUTE_VIEWPOINTS:
        if out["viewpoint_scores"][vp] <= 0:
            out["viewpoint_scores"][vp] = out["next_landmark_prob"][vp]

    prob_sum = float(sum(out["next_landmark_prob"].values()))
    if prob_sum > 1e-6:
        for vp in _ROUTE_VIEWPOINTS:
            out["next_landmark_prob"][vp] = float(out["next_landmark_prob"][vp] / prob_sum)
    else:
        score_sum = float(sum(out["viewpoint_scores"].values()))
        if score_sum > 1e-6:
            for vp in _ROUTE_VIEWPOINTS:
                out["next_landmark_prob"][vp] = float(out["viewpoint_scores"][vp] / score_sum)

    return out



def prompt_updator_v2(original_prompt, ongoing_task=None, action_code=None, observations=None, action_seq_num=1):
    ori_prompt_splits = original_prompt.split("\n\n")

    intro_text = ori_prompt_splits[0]
    action_space_text = ori_prompt_splits[1]
    obs_direc_text = ori_prompt_splits[2]
    navi_instruction_text = ori_prompt_splits[3]
    action_obs_text = ori_prompt_splits[4]
    action_predict_prompt = ori_prompt_splits[5]


    # action_seq = "\n".join(ori_prompt_splits[14:-3])
    action_obs_seq = action_obs_text.split("\n")
    action_obs_seq = action_obs_seq[1:]

    if len(action_obs_seq) > 0:
        pattern = re.compile(r'^\d+')
        action_seq_num = 0
        for i in range(len(action_obs_seq) - 1, -1, -1):
            m = re.match(pattern, action_obs_seq[i])
            if m:
                action_seq_num = int(m.group()) + 1
                break
        if not action_seq_num:
            action_seq_num = 1
        action_obs_seq_text = "\n".join(action_obs_seq)
    else:
        action_obs_seq_text = ""


    action_str = ""
    if action_code == 0:
        action_str = "STOP"
    elif action_code == 1:
        action_str = "MOVE FORWARD"
    elif action_code == 2:
        action_str = "TURN LEFT"
    elif action_code == 3:
        action_str = "TURN RIGHT"
    elif action_code == 4:
        action_str = "GO UP"
    elif action_code == 5:
        action_str = "GO DOWN"

    if action_str != "":
        if action_obs_seq_text != "":
            action_obs_seq_text = action_obs_seq_text+"\n"+f"{action_seq_num}. {action_str}"
        else:
            action_obs_seq_text = f"{action_seq_num}. {action_str}"

    observation_str = ""
    if observations:
        for landmark in observations:
            coarse_grained_loc, fine_grained_loc = observations[landmark]
            landmark_obs_str = f"There is {landmark} on the {fine_grained_loc} side of your {coarse_grained_loc} view.\n"
            observation_str += landmark_obs_str
    observation_str = observation_str.strip("\n")

    if observation_str != "":
        if action_obs_seq_text != "":
            action_obs_seq_text = action_obs_seq_text + "\n" + observation_str
        else:
            action_obs_seq_text = action_obs_seq_text + observation_str

    action_obs_text = "Action Sequence:\n"+action_obs_seq_text

    prompt = f"""{intro_text}

{action_space_text}

{obs_direc_text}

{navi_instruction_text}

{action_obs_text}

{action_predict_prompt}"""

    return prompt


def action_parser(action_str):
    action_str = action_str.lower()
    action_code = -1
    if "stop" in action_str:
        action_code = 0
    elif "forward" in action_str or "move forward" in action_str:
        action_code = 1
    elif "left" in action_str or "turn left" in action_str:
        action_code = 2
    elif "right" in action_str or "turn right" in action_str:
        action_code = 3
    elif "up" in action_str or "go up" in action_str:
        action_code = 4
    elif "down" in action_str or "go down" in action_str:
        action_code = 5

    return action_code

