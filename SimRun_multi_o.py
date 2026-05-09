
import json
import os
import copy
import torch
import time
import airsim
from airsim import ImageRequest, ImageType
import pickle
import numpy as np

# typing (needed early for port helpers)
from typing import List, Optional


# ==================== AirSim raw client helper ====================
def _read_airsim_port_from_settings(default_port: int = 30001) -> int:
    """Try to read ApiServerPort from settings.json.

    AirVLN/CityNavAgent often runs AirSim on a **non-default** port (e.g., 30001).
    If we can't find settings.json, we fall back to default_port (default=30001).
    """
    candidates: List[str] = []

    # 1) settings.json next to this script
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(here, "settings.json"))
    except Exception:
        pass

    # 2) settings.json in current working dir
    candidates.append(os.path.join(os.getcwd(), "settings.json"))

    # 3) AirSim default locations (Linux)
    home = os.path.expanduser("~")
    candidates.append(os.path.join(home, "Documents", "AirSim", "settings.json"))
    candidates.append(os.path.join(home, ".config", "AirSim", "settings.json"))
    candidates.append(os.path.join(home, ".airsim", "settings.json"))

    # 4) explicit env hint
    env_p = os.environ.get("AIRSIM_SETTINGS_PATH", "")
    if env_p:
        candidates.append(env_p)

    for p in candidates:
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    s = json.load(f)
                port = int(s.get("ApiServerPort", default_port))
                return port
        except Exception:
            continue
    return int(default_port)


def _extract_airsim_ports_from_run_ret(run_ret) -> List[int]:
    """Best-effort extraction of AirSim ApiServerPort(s) from AirVLN tool return.

    In your logs, AirVLNSimulatorClientTool prints something like:
      [True, ['127.0.0.1', [30001]]]
    We try to parse this structure (and nested variants).
    """
    ports: List[int] = []

    def _walk(x):
        nonlocal ports
        if x is None:
            return
        # Pattern: [ip_str, [port_int, ...]]
        if isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], str) and isinstance(x[1], (list, tuple)):
            if all(isinstance(p, int) for p in x[1]):
                ports.extend(list(x[1]))
                return
        if isinstance(x, dict):
            for v in x.values():
                _walk(v)
        elif isinstance(x, (list, tuple)):
            for it in x:
                _walk(it)

    _walk(run_ret)

    # de-dup, keep order
    uniq: List[int] = []
    seen = set()
    for p in ports:
        if p not in seen:
            uniq.append(int(p))
            seen.add(p)
    # AirVLN often uses 30001+; filter out obvious command ports like 30000 if present
    uniq = [p for p in uniq if p != 30000]
    return uniq


def _make_raw_airsim_client(ip: str = "127.0.0.1", port=None, port_candidates: Optional[List[int]] = None) -> "airsim.MultirotorClient":
    """Create an AirSim MultirotorClient for low-level multi-vehicle calls.

    Key fix for your error:
      - AirVLN/CityNavAgent often runs AirSim on **30001**, not the default 414从.
      - If we connect to the wrong port, msgpackrpc session stays None, causing:
            AttributeError: 'NoneType' object has no attribute 'send_message'
    """
    candidates: List[int] = []
    if port is not None:
        candidates.append(int(port))
    if port_candidates:
        candidates.extend([int(p) for p in port_candidates if p is not None])

    # settings.json (various locations)
    candidates.append(_read_airsim_port_from_settings())
    # common fallbacks
    candidates.extend([30001, 41451])

    # de-dup, keep order
    uniq: List[int] = []
    seen = set()
    for p in candidates:
        if p not in seen:
            uniq.append(int(p))
            seen.add(p)

    def _quick_check(client_obj) -> bool:
        try:
            if hasattr(client_obj, "ping"):
                client_obj.ping()
                return True
            if hasattr(client_obj, "getServerVersion"):
                _ = client_obj.getServerVersion()
                return True
            # fallback (might retry internally)
            client_obj.confirmConnection()
            return True
        except Exception:
            return False

    last_exc: Optional[Exception] = None
    for p in uniq:
        try:
            c = airsim.MultirotorClient(ip=ip, port=int(p))
            if _quick_check(c):
                print(f"[AirSim] raw client connected: {ip}:{p}")
                return c
        except Exception as e:
            last_exc = e
            continue

    # As a last resort, return the default client and let later calls print meaningful errors.
    fallback_port = uniq[0] if uniq else 30001
    c = airsim.MultirotorClient(ip=ip, port=int(fallback_port))
    try:
        c.confirmConnection()
    except Exception as e:
        last_exc = e
    print(f"[WARN] AirSim raw client failed to connect. Tried ports={uniq}. Last error={last_exc}")
    return c
import cv2
import os
import sys
import pandas as pd  # 数据分析核心库
from datetime import datetime
from tqdm import tqdm
import argparse
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# --- 路径与环境设置 ---
sys.path.append(os.path.abspath("external/Grounded_Sam_Lite"))
from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool
from airsim_plugin.airsim_settings import ObservationDirections
from evaluator.nav_evaluator import CityNavEvaluator
from src.llm.query_llm import OpenAI_LLM_v2
from utils.env_utils import get_pano_observations
from external.Grounded_Sam_Lite.grounded_sam_api import GroundedSam
from external.Grounded_Sam_Lite.groundingdino.util.inference import load_model

from torch import amp
from PIL import Image

from src.llm.prompt_builder import route_planning_prompt_builder, parse_viewpoint_response_v2
from utils.env_utils import getPoseAfterMakeActions
from utils.maps import convert_global_pc, statistical_filter, find_closest_node
from utils.utils import calculate_movement_steps, calculate_movement_steps_mem

from external.lm_nav.navigation_graph import NavigationGraph
from external.lm_nav import pipeline
from scipy.spatial.transform import Rotation as R

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
warnings.filterwarnings("ignore", message=".*use_reentrant.*")
warnings.filterwarnings("ignore", message=".*requires_grad=True.*")
warnings.filterwarnings("ignore", message=".*autocast.*")
warnings.filterwarnings("ignore", message="Failed to load custom C\\+\\+ ops.*")
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated.*")

# ==================== 1) 多机控制 Wrapper ====================

class AgentSpecificToolWrapper:
    """
    解决 AirSim Python API 默认只能控制单机的限制。
    通过拦截调用，强制注入 vehicle_name。
    """
    class ClientWrapper:
        """内部类：用于封装 raw_client，拦截 simGetVehiclePose 等底层调用"""
        def __init__(self, raw_client, agent_name):
            self.raw_client = raw_client
            self.agent_name = agent_name

        def simGetVehiclePose(self, vehicle_name=""):
            target = vehicle_name if vehicle_name else self.agent_name
            return self.raw_client.simGetVehiclePose(vehicle_name=target)

        def __getattr__(self, name):
            return getattr(self.raw_client, name)

    def __init__(self, original_tool, agent_name, raw_client):
        self.orig = original_tool
        self.agent_name = agent_name
        # 关键：封装 client，确保通过 tool.client 访问时也能注入 vehicle_name
        self.client = self.ClientWrapper(raw_client, agent_name)

    def getImageResponses_v2(self, camera_id='front_0', **kwargs):
        """
        返回结构尽量兼容旧版 get_pano_observations 的期望格式：[[[rgb, depth_norm]]]
        """
        requests = [
            airsim.ImageRequest(camera_id, airsim.ImageType.Scene, False, False),
            airsim.ImageRequest(camera_id, airsim.ImageType.DepthPerspective, True, False),
        ]
        responses = self.client.raw_client.simGetImages(requests, vehicle_name=self.agent_name)

        # RGB
        rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        rgb = rgb.reshape(responses[0].height, responses[0].width, 3)

        # Depth (float, meters) -> normalize to 0~1 (match original pipeline expectation)
        depth = np.array(responses[1].image_data_float, dtype=np.float32)
        depth = depth.reshape(responses[1].height, responses[1].width)

        MAX_D = 300.0
        depth = np.clip(depth / MAX_D, 0.0, 1.0)

        processed = [rgb, depth]
        return [[processed]]

    def setPose(self, pose):
        """Set pose for this agent.

        Accepts:
        - airsim.Pose
        - 7D list/tuple/np.ndarray: [x,y,z,qx,qy,qz,qw]
        Always casts to Python floats to avoid msgpackrpc numpy serialization issues.
        """
        try:
            pose2 = convert_airsim_pose(pose)
            self.client.raw_client.simSetVehiclePose(pose2, True, vehicle_name=self.agent_name)
        except Exception as e:
            print(f"[{self.agent_name}] Pose Error: {e}")

    def setPoses(self, poses):
        """Compat: tool.setPoses([[pose]]) / tool.setPoses([pose]) / tool.setPoses(pose)."""
        try:
            if poses is None:
                return
            if isinstance(poses, list) and len(poses) > 0:
                inner = poses[0]
                pose = inner[0] if isinstance(inner, list) else inner
            else:
                pose = poses
            self.setPose(pose)
        except Exception as e:
            print(f"[{self.agent_name}] Pose Error: {e}")

    def __getattr__(self, name):
        return getattr(self.orig, name)

def convert_airsim_pose(pose):
    """Convert a 7D pose list/array to airsim.Pose **with Python floats**.

    IMPORTANT: msgpackrpc cannot serialize numpy.float32. If numpy scalars leak into
    AirSim RPC calls, you'll see:
        'numpy.float32' object has no attribute 'to_msgpack'
    """
    if isinstance(pose, airsim.Pose):
        return pose
    pose = list(pose)
    if len(pose) < 7:
        raise ValueError(f"convert_airsim_pose expects 7 values, got {len(pose)}")
    x, y, z, qx, qy, qz, qw = [float(p) for p in pose[:7]]
    return airsim.Pose(
        position_val=airsim.Vector3r(x, y, z),
        orientation_val=airsim.Quaternionr(x_val=qx, y_val=qy, z_val=qz, w_val=qw),
    )




def quat_to_list(q: airsim.Quaternionr) -> List[float]:
    return [float(q.x_val), float(q.y_val), float(q.z_val), float(q.w_val)]
# ==================== 2) 指标记录器（可选） ====================

class ExperimentLogger:
    def __init__(self):
        self.records = []

    def log_episode(self, mode, episode_id, success, steps, path_length, vlm_calls, sync_count, metrics=None):
        record = {
            "Mode": mode,
            "Episode": episode_id,
            "Success": 1 if success else 0,
            "Steps": steps,
            "Path_Length": path_length,
            "VLM_Cost": vlm_calls,
            "Sync_Events": sync_count,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        if metrics:
            record.update(metrics)
        self.records.append(record)

    def save_to_csv(self, filename="experiment_results.csv"):
        df = pd.DataFrame(self.records)
        df.to_csv(filename, index=False)
        print(f"\n[Data] Results saved to {filename}")
        if len(df) > 0 and "Mode" in df.columns:
            summary = df.groupby("Mode")[["Success", "Steps", "Path_Length", "VLM_Cost"]].mean()
            print("\n=== Experiment Summary (Mean Values) ===")
            print(summary)
            print("========================================\n")
        return df

# ==================== 3) 多无人机协同：黑板 + 置信度 ====================

@dataclass
class CoNavConfig:
    # swarm init
    num_uavs: int = 2
    airsim_port: Optional[int] = None  # AirSim ApiServerPort (default read from settings.json)
    spawn_sep_m: float = 5.0
    spawn_axis: str = "z"          # "y" 或 "z"
    takeoff_steps: int = 5         # 与单机一致的“起飞动作步数”
    # collaboration
    enable_blackboard: bool = True
    enable_verification: bool = True
    enable_regroup: bool = True
    enable_search_during_regroup: bool = True
    freeze_executor_on_found: bool = True
    # thresholds
    assoc_dist_m: float = 20.0          # executor vs verifier position closeness
    conf_accept: float = 0.5          # scaled confidence threshold
    # hypothesis quality gates (avoid infinite verify on empty/noisy masks)
    min_hyp_points: int = 120         # require at least N points to post a blackboard hypothesis
    min_vlm_raw_for_hyp: float = 0.2 # require minimal VLM score (raw) to post
    require_measurement_for_found: bool = True  # if LLM says found but no measurement, keep exploring (no freeze)
    enable_fallback_motion: bool = True         # if no measurement, do a small motion toward LLM-chosen direction
    fallback_step_m: float = 6.0                # meters per fallback move
    # move behavior when LLM has NOT found the landmark but provides a related proxy target
    direct_fly_to_related: bool = False          # True: fly directly to VLM pointcloud center (single-like), False: smooth step
# meters per fallback move
    # semantic fusion weights
    sem_w_vlm: float = 0.8
    sem_w_llm: float = 0.2
    conf_min_to_post: float = 0.15     # too low -> still post, but as weak hypothesis
    max_verify_attempts: int = 2
    # camera/view planning
    camera_fov_deg: float = 90.0
    view_margin_m: float = 2.0
    view_standoff_min_m: float = 5.0
    view_standoff_max_m: float = 15.0
    view_height_gain: float = 0.25     # additional height relative to pc radius
    # ablation style knobs
    use_llm_in_conf: bool = True
    use_vlm_in_conf: bool = True
    # logging
    verbose: bool = True

class TeamBlackboard:
    """
    共享黑板：存放当前 next_landmark 的候选假设。
    结构尽量简单，方便后续做消融。
    """
    def __init__(self):
        self.pending: Optional[Dict[str, Any]] = None   # one pending hypothesis for current landmark
        self.history: List[Dict[str, Any]] = []

    def post(self, hyp: Dict[str, Any]):
        self.pending = hyp
        self.history.append(copy.deepcopy(hyp))

    def clear(self):
        self.pending = None

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def scale_vlm_score(raw: float, mid: float = 0.3, sharpness: float = 10.0) -> float:
    """
    VLM raw score 通常 >0.35 可算“较高”，这里做一个平滑映射到 [0,1]
    raw=0.35 -> 0.5
    """
    raw = float(raw)
    return float(_sigmoid(sharpness * (raw - mid)))

def combine_semantic_conf(is_found_llm: bool, vlm_raw: float, cfg: CoNavConfig, npts: int = 0) -> float:
    """Fuse LLM(is_found) + VLM(raw score) into a semantic confidence in [0,1].

    - VLM raw score is scaled with a logistic around 0.35 ("high").
    - Weighted fusion: sem = w_vlm * scaled_vlm + w_llm * is_found_llm.
    - If there is NO pointcloud measurement (npts==0), confidence is forced low to
      prevent meaningless verification loops.
    """
    if npts <= 0:
        return 0.0 if cfg.require_measurement_for_found else (float(cfg.sem_w_llm) if is_found_llm else 0.0)

    v = scale_vlm_score(vlm_raw) if vlm_raw is not None else 0.0
    l = 1.0 if is_found_llm else 0.0
    wv = float(cfg.sem_w_vlm)
    wl = float(cfg.sem_w_llm)
    s = wv * v + wl * l
    return float(max(0.0, min(1.0, s)))

def pc_center_radius(pc: np.ndarray) -> Tuple[np.ndarray, float]:
    if pc is None or len(pc) == 0 or (len(pc) == 1 and not np.any(pc)):
        return np.zeros(3, dtype=np.float32), 0.0
    c = np.mean(pc, axis=0).astype(np.float32)
    r = float(np.max(np.linalg.norm(pc - c[None, :], axis=1)))
    return c, r

def compute_view_pose_to_cover_pc(
    pc: np.ndarray,
    target_center: np.ndarray,
    executor_pose: airsim.Pose,
    cfg: CoNavConfig,
) -> airsim.Pose:
    """
    给定点云(pc)与中心(target_center)，基于相机 FOV 计算 verifier 的观察位姿，
    确保点云整体落入视野（近似用包围球半径 r）。
    """
    _, r = pc_center_radius(pc)
    theta = math.radians(cfg.camera_fov_deg * 0.5)
    # 让半径 r 的球体落入视野：d >= r / tan(theta) + margin
    d = (r / max(1e-6, math.tan(theta))) + cfg.view_margin_m
    d = float(np.clip(d, cfg.view_standoff_min_m, cfg.view_standoff_max_m))

    ex = np.array([executor_pose.position.x_val, executor_pose.position.y_val], dtype=np.float32)
    c2 = np.array([target_center[0], target_center[1]], dtype=np.float32)

    v = ex - c2
    n = float(np.linalg.norm(v))
    if n < 1e-3:
        v = np.array([1.0, 0.0], dtype=np.float32)
        n = 1.0
    u = v / n

    # verifier 放在 “中心朝 executor 方向” 的 standoff 距离处，朝向中心
    pos_xy = c2 + u * d

    # 高度：在中心高度基础上往上抬一些（AirSim NED: z 越负越高）
    z = float(target_center[2] - cfg.view_height_gain * r)
    # 任务里通常希望飞行高度不低于 -2m（更高 => z 更负）
    z = min(z, -2.0)

    # yaw：指向中心
    yaw = math.atan2(c2[1] - pos_xy[1], c2[0] - pos_xy[0])
    quat = R.from_euler('z', yaw, degrees=False).as_quat()  # x,y,z,w
    return convert_airsim_pose([float(pos_xy[0]), float(pos_xy[1]), z, float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])])

# ==================== 4) VLM 多候选分割（用于 verifier 选择最近实例） ====================

def vlm_candidate_masks(
    vlm: Any,
    rgb_bgr: np.ndarray,
    text_prompt: str,
    max_candidates: int = 6,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> List[Tuple[np.ndarray, float, str]]:
    """
    返回多个候选 mask，用于“多个目标时选择最近 executor 提出的那个”。
    如果 vlm 不是 GroundedSam，则退化为 greedy 的单个候选。
    """
    # fallback: greedy
    if not hasattr(vlm, "get_dino_output") or not hasattr(vlm, "sam") or not hasattr(vlm, "transform"):
        res = vlm.greedy_mask_predict(rgb_bgr, text_prompt, visualize=False)
        if isinstance(res, tuple) and len(res) == 4:
            mask, ok, score, phrase = res
        elif isinstance(res, tuple) and len(res) == 2:
            mask, ok = res
            score, phrase = 0.0, ""
        else:
            return []
        return [(mask.astype(bool), float(score), str(phrase))] if ok else []

    h, w = rgb_bgr.shape[:2]
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb)
    image_tensor, _ = vlm.transform(image_pil, None)

    boxes_filt, pred_phrases, scores_filt = vlm.get_dino_output(
        image_tensor, text_prompt, box_threshold=box_threshold, text_threshold=text_threshold, with_logits=False
    )
    if boxes_filt is None or len(pred_phrases) == 0:
        return []

    # 选择“最先匹配到的短语”作为 best_phrase
    in_phrases = [p.strip(" ") for p in text_prompt.split(".") if p.strip(" ")]
    best_phrase = None
    for inp in in_phrases:
        for pp in pred_phrases:
            if inp in pp:
                best_phrase = inp
                break
        if best_phrase is not None:
            break
    if best_phrase is None:
        return []

    # 收集所有包含 best_phrase 的 box
    cand = []
    for i, pp in enumerate(pred_phrases):
        if best_phrase in pp:
            cand.append((boxes_filt[i:i+1, :].cpu(), float(scores_filt[i].item()), best_phrase))
    if len(cand) == 0:
        return []

    # top-k by score
    cand.sort(key=lambda x: -x[1])
    cand = cand[:max_candidates]

    boxes = torch.cat([c[0] for c in cand], dim=0)  # (k,4) cxcywh normalized
    scores = [c[1] for c in cand]
    phrases = [c[2] for c in cand]

    # convert to xyxy in pixel space
    boxes_xyxy = boxes.clone()
    boxes_xyxy = boxes_xyxy * torch.tensor([w, h, w, h], dtype=boxes_xyxy.dtype)
    boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
    boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]

    vlm.sam.set_image(rgb)
    transformed_boxes = vlm.sam.transform.apply_boxes_torch(boxes_xyxy, rgb.shape[:2]).to(vlm.device)

    masks, _, _ = vlm.sam.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    masks = masks.detach().cpu().numpy()  # (k,1,H,W)

    out = []
    for i in range(masks.shape[0]):
        m = masks[i, 0].astype(bool)
        out.append((m, float(scores[i]), str(phrases[i])))
    return out

# ==================== 5) 单机 explore pipeline（轻改：返回 pc / 置信度 / 可冻结移动） ====================

def explore_pipeline_by_sam(
    curr_pose,
    llm, vlm,
    cfg: CoNavConfig,
    image_path: List[str],
    rgb_imgs: List[np.ndarray],
    dep_imgs: List[np.ndarray],
    obs_poses: List[np.ndarray],
    navigation_instruction: str,
    scene_objects: List[str],
    landmarks_route: List[str],
    next_landmark_idx: int,
    freeze_on_found: bool = False,
):
    """
    复用单机流程：
      1) LLM route planning -> is_found + (viewpoint->object)
      2) VLM grounding+seg -> mask -> point cloud -> 3D 观测(mu/cov/conf)
      3) low-level 生成 new_pose (可选：发现地标后冻结，不移动)

    返回：
      step_size, new_pose, next_landmark_found,
      obs_mu(3,), cov_xy(2,2), obs_conf(0~1),
      pc_bundle(dict: pc/points/vp/obj/vlm_score/llm_found)
    """
    obs_viewpoint = ["left", "slightly_left", "front", "slightly_right", "right"]
    viewpoint_img_path = {}
    viewpoint_rgb_imgs = {}
    viewpoint_dep_imgs = {}
    viewpoint_poses = {}
    next_subgoal_found = False

    for k in range(len(obs_viewpoint)):
        viewpoint = obs_viewpoint[k]
        viewpoint_img_path[viewpoint] = image_path[k]
        viewpoint_rgb_imgs[viewpoint] = rgb_imgs[k]
        viewpoint_dep_imgs[viewpoint] = dep_imgs[k]
        viewpoint_poses[viewpoint] = obs_poses[k]

    traversed_landmarks = landmarks_route[:next_landmark_idx]
    route_predict_prompt = route_planning_prompt_builder(
        navigation_instruction, landmarks_route, traversed_landmarks, landmarks_route[next_landmark_idx]
    )

    route_predicted = llm.query_viewpoint_api(route_predict_prompt, viewpoint_img_path, show_response=False)

    try:
        route_predicted_dict = parse_viewpoint_response_v2(route_predicted)
    except Exception as e:
        print(f"[⚠️ Warning] JSON Parse Failed. Error: {e}")
        print(f"[⚠️ content] LLM Response: {route_predicted}")
        cur_pos_fallback = np.array([curr_pose.position.x_val, curr_pose.position.y_val, curr_pose.position.z_val], dtype=np.float32)
        cov_fallback = np.array([[200.0, 0.0], [0.0, 200.0]], dtype=np.float32)
        pc_bundle = {"pc": np.zeros((1, 3), dtype=np.float32), "npts": 0, "vp": None, "obj": None,
                     "vlm_raw": 0.0, "llm_found": False}
        return 0, curr_pose, False, cur_pos_fallback, cov_fallback, 0.0, pc_bundle

    if route_predicted_dict.get("is_found", False):
        next_subgoal_found = True

    # --- build semantic point cloud candidates ---
    candidates = []  # (vp, obj, pc, n_points, vlm_raw)
    seg_succ_all = False

    for vp, obj in route_predicted_dict.items():
        if vp == "is_found":
            continue
        rgb_img = viewpoint_rgb_imgs[vp]
        dep_img = viewpoint_dep_imgs[vp].squeeze()
        pose = viewpoint_poses[vp]

        # robust unpack
        res = vlm.greedy_mask_predict(rgb_img, obj, visualize=False)
        if isinstance(res, tuple) and len(res) == 4:
            route_mask, seg_succ, vlm_raw, _phrase = res
        elif isinstance(res, tuple) and len(res) == 2:
            route_mask, seg_succ = res
            vlm_raw = 0.0
        else:
            route_mask, seg_succ, vlm_raw = np.zeros_like(dep_img, dtype=bool), False, 0.0

        seg_succ_all = seg_succ_all or seg_succ
        if not seg_succ:
            continue

        part_pc, filter_idx = convert_global_pc(dep_img, 90, pose, route_mask)
        semantic_part_pc = part_pc[filter_idx]

        if len(semantic_part_pc) > 30:
            semantic_part_pc, _ = statistical_filter(semantic_part_pc)

        if len(semantic_part_pc) > 0:
            candidates.append((vp, obj, semantic_part_pc, int(len(semantic_part_pc)), float(vlm_raw)))

    # deterministic pick: most points, then fixed viewpoint priority
    best_npts = 0
    best_vp, best_obj, semantic_pc, best_vlm_raw = None, None, np.zeros((1, 3), dtype=np.float32), 0.0

    if len(candidates) > 0:
        vp_priority = {"front": 0, "slightly_right": 1, "slightly_left": 2, "right": 3, "left": 4}
        candidates.sort(key=lambda x: (-x[3], vp_priority.get(x[0], 99)))
        best_vp, best_obj, semantic_pc, best_npts, best_vlm_raw = candidates[0]
        print(f"[PC select] use {best_vp}/{best_obj}, points={best_npts}, vlm_raw={best_vlm_raw:.3f}")

    # choose a "preferred" viewpoint from LLM output (first non-unknown), used for fallback motion
    preferred_vp = None
    preferred_obj = None
    for vp, obj in route_predicted_dict.items():
        if vp == "is_found":
            continue
        if isinstance(obj, str) and obj.lower() != "unknown":
            preferred_vp, preferred_obj = vp, obj
            break

# --- Belief observation (mu/cov/conf) ---
    cur_pos = np.array([curr_pose.position.x_val, curr_pose.position.y_val, curr_pose.position.z_val], dtype=np.float32)

    if seg_succ_all and semantic_pc is not None and len(semantic_pc) >= 1 and np.any(semantic_pc):
        obs_mu = np.mean(semantic_pc, axis=0).astype(np.float32)
        if len(semantic_pc) >= 10:
            cov_xy = np.cov(semantic_pc[:, :2].T).astype(np.float32)
        else:
            cov_xy = np.array([[100.0, 0.0], [0.0, 100.0]], dtype=np.float32)
        # 用点数映射到 [0,1] 作为几何置信度（和语义置信度分开）
        geom_conf = float(min(1.0, best_npts / 150.0))
    else:
        obs_mu = cur_pos.copy()
        cov_xy = np.array([[200.0, 0.0], [0.0, 200.0]], dtype=np.float32)
        geom_conf = 0.0

    # 运动目标：默认会向 obs_mu 平滑移动；
    # 多机协同时：仅当 “LLM found + 有测量” 时才冻结 executor（避免空mask导致卡死）。
    has_measurement = bool(seg_succ_all and semantic_pc is not None and len(semantic_pc) >= 1 and np.any(semantic_pc))

    # If LLM says found but no measurement, we keep exploring (optionally do a small fallback motion)
    if next_subgoal_found and (not has_measurement) and cfg.require_measurement_for_found:
        next_subgoal_found = False  # treat as not-found for team logic

    if not has_measurement:
        # fallback: move a bit toward the LLM-preferred viewpoint to change observation, instead of staying still
        if cfg.enable_fallback_motion and preferred_vp is not None:
            q = curr_pose.orientation
            yaw0 = R.from_quat([float(q.x_val), float(q.y_val), float(q.z_val), float(q.w_val)]).as_euler("zyx")[0]
            delta = {"left": math.radians(60), "slightly_left": math.radians(30),
                     "front": 0.0, "slightly_right": -math.radians(30), "right": -math.radians(60)}.get(preferred_vp, 0.0)
            yaw1 = yaw0 + delta
            step = float(cfg.fallback_step_m)
            route_coords = cur_pos.copy()
            route_coords[0] = float(route_coords[0] + step * math.cos(yaw1))
            route_coords[1] = float(route_coords[1] + step * math.sin(yaw1))
        else:
            route_coords = cur_pos
    else:
        # measurement available -> move toward observation center (proxy target)
        if freeze_on_found and next_subgoal_found:
            route_coords = cur_pos
        else:
            route_coords = obs_mu.copy()
            if not np.any(route_coords) or np.any(np.isnan(route_coords)):
                route_coords = cur_pos

            # NEW: if landmark is NOT found yet, but LLM suggested a related object and we
            # have a valid VLM pointcloud center, directly fly to that proxy target.
            if (not next_subgoal_found) and getattr(cfg, "direct_fly_to_related", True):
                pass
            else:
                # legacy smoothing (ablation)
                alpha = 0.6
                route_coords = alpha * route_coords + (1 - alpha) * cur_pos

            if route_coords[2] > -2:
                route_coords[2] = -2

    rel_trans = route_coords - cur_pos
    yaw = float(np.arctan2(rel_trans[1], rel_trans[0])) if (abs(rel_trans[0]) + abs(rel_trans[1]) > 1e-6) else 0.0
    new_quat = R.from_euler('z', yaw, degrees=False).as_quat()
    new_pose = convert_airsim_pose([float(v) for v in list(route_coords)] + [float(v) for v in list(new_quat)])

    dist = np.abs(rel_trans)
    step_size = np.abs(np.rad2deg(yaw)) // 15 + dist[2] // 2 + np.sqrt(dist[0]**2 + dist[1]**2) // 5

    pc_bundle = {
        "pc": semantic_pc.astype(np.float32),
        "npts": int(best_npts),
        "vp": best_vp,
        "obj": best_obj,
        "vlm_raw": float(best_vlm_raw),
        "llm_found": bool(next_subgoal_found),
        "geom_conf": float(geom_conf),
    }

    return int(step_size), new_pose, next_subgoal_found, obs_mu, cov_xy, geom_conf, pc_bundle

# ==================== 6) verifier 流程：只用前方三视角 + 最近实例选择 ====================

def _verify_prompt(landmark_name: str) -> str:
    # 输出格式与 parse_viewpoint_response_v2 兼容
    return f"""
You are helping a drone verify a landmark hypothesis.
Given 3 front-facing views (slightly_left, front, slightly_right), answer whether the landmark <{landmark_name}> appears.

Output JSON only:
If you can see it in one view:
{{
  "is_found": true,
  "front": "{landmark_name}"
}}
(or use "slightly left"/"slightly right" accordingly)

If you cannot see it:
{{
  "is_found": false,
  "front": "unknown",
  "slightly left": "unknown",
  "slightly right": "unknown"
}}
"""

def verifier_verify_hypothesis(
    verifier_name: str,
    verifier_tool: AgentSpecificToolWrapper,
    verifier_pose: airsim.Pose,
    executor_pose: airsim.Pose,
    hyp: Dict[str, Any],
    llm, vlm,
    scene_id: int,
    cfg: CoNavConfig,
) -> Dict[str, Any]:
    """
    1) 飞到 view_pose（保证看全点云）
    2) 仅用 slightly_left/front/slightly_right 三个视角
    3) LLM 判断是否存在该地标
    4) VLM 以地标名分割；若多实例，选离 executor 提案最近的实例
    """
    target_center = hyp["pos_mu"].astype(np.float32)
    pc = hyp.get("pc", np.zeros((1, 3), dtype=np.float32))
    view_pose = compute_view_pose_to_cover_pc(pc, target_center, executor_pose, cfg)

    # move verifier to view pose (teleport style, consistent with existing pipeline)
    verifier_tool.setPoses([[view_pose]])
    verifier_pose = view_pose

    # observe
    pano_obs, pano_pose = get_pano_observations(verifier_pose, verifier_tool, scene_id=scene_id)

    # mapping (与 explore 保持一致)
    # left=6, slightly_left=7, front=0, slightly_right=1, right=2, back=4 (此处只用三视角)
    idx_map = {"slightly_left": 7, "front": 0, "slightly_right": 1}
    obs_imgs = {k: pano_obs[idx_map[k]][0] for k in idx_map}
    obs_deps = {k: pano_obs[idx_map[k]][1] for k in idx_map}
    obs_poses = {k: pano_pose[idx_map[k]] for k in idx_map}

    # save images for llm api
    os.makedirs("obs_imgs", exist_ok=True)
    img_paths = {}
    for k, img in obs_imgs.items():
        p = f"obs_imgs/{verifier_name}_verify_{k}.png"
        cv2.imwrite(p, img)
        img_paths[k] = p

    # LLM verify
    llm_prompt = _verify_prompt(hyp["landmark_name"])
    llm_resp = llm.query_viewpoint_api(llm_prompt, img_paths, show_response=False)

    try:
        llm_dict = parse_viewpoint_response_v2(llm_resp)
        llm_found = bool(llm_dict.get("is_found", False))
    except Exception as e:
        print(f"[{verifier_name}][verify] JSON Parse Failed: {e}")
        print(f"[{verifier_name}][verify] LLM Response: {llm_resp}")
        llm_dict = {"is_found": False}
        llm_found = False

    # VLM segmentation (multi-candidate per view) + choose closest to executor hyp center
    best = None  # (dist, view, pc, npts, vlm_raw)
    for view in ["front", "slightly_left", "slightly_right"]:
        rgb = obs_imgs[view]
        dep = obs_deps[view].squeeze()
        pose = obs_poses[view]

        cands = vlm_candidate_masks(vlm, rgb, hyp["landmark_name"])
        for (mask, vlm_raw, phrase) in cands:
            part_pc, filter_idx = convert_global_pc(dep, 90, pose, mask)
            semantic_pc = part_pc[filter_idx]
            if len(semantic_pc) > 30:
                semantic_pc, _ = statistical_filter(semantic_pc)
            if len(semantic_pc) == 0:
                continue

            mu = np.mean(semantic_pc, axis=0).astype(np.float32)
            dist = float(np.linalg.norm(mu[:2] - target_center[:2]))
            cand_tuple = (dist, view, semantic_pc, int(len(semantic_pc)), float(vlm_raw), mu)
            if (best is None) or (cand_tuple[0] < best[0]):
                best = cand_tuple

    if best is None:
        v_pc = np.zeros((1, 3), dtype=np.float32)
        v_mu = np.array([verifier_pose.position.x_val, verifier_pose.position.y_val, verifier_pose.position.z_val], dtype=np.float32)
        v_npts = 0
        v_vlm_raw = 0.0
    else:
        _, best_view, v_pc, v_npts, v_vlm_raw, v_mu = best

    v_conf = combine_semantic_conf(llm_found, v_vlm_raw, cfg, npts=int(v_npts))

    out = {
        "verifier": verifier_name,
        "view_pose": view_pose,
        "llm_found": llm_found,
        "vlm_raw": float(v_vlm_raw),
        "sem_conf": float(v_conf),
        "pos_mu": v_mu.astype(np.float32),
        "pc": v_pc.astype(np.float32),
        "npts": int(v_npts),
    }
    return out

# ==================== 7) 多无人机主循环 ====================

def _spawn_swarm_poses(base_pose: airsim.Pose, cfg: CoNavConfig) -> List[airsim.Pose]:
    """
    在出发点附近部署多架无人机：
      - 默认沿 y 方向每 5m 间隔（更符合“平面垂直方向”直觉）
      - 可切换为沿 z(高度)方向间隔
    """
    poses = []
    x0, y0, z0 = base_pose.position.x_val, base_pose.position.y_val, base_pose.position.z_val
    q = base_pose.orientation
    # centered offsets
    for i in range(cfg.num_uavs):
        off = (i - (cfg.num_uavs - 1) / 2.0) * cfg.spawn_sep_m
        x, y, z = float(x0), float(y0), float(z0)
        if cfg.spawn_axis.lower() == "y":
            y = float(y0 + off)
        elif cfg.spawn_axis.lower() == "z":
            z = float(z0 - abs(off))  # NED: more negative => higher
        poses.append(airsim.Pose(airsim.Vector3r(x, y, z), q))
    return poses

def _takeoff_like_single(pose: airsim.Pose, steps: int) -> airsim.Pose:
    cur = pose
    for _ in range(int(steps)):
        cur = getPoseAfterMakeActions(cur, [4])  # GO_UP
    return cur

def _dist_xy(a: airsim.Pose, b_xyz: np.ndarray) -> float:
    ax = np.array([a.position.x_val, a.position.y_val], dtype=np.float32)
    bx = np.array([b_xyz[0], b_xyz[1]], dtype=np.float32)
    return float(np.linalg.norm(ax - bx))

def _choose_verifier(agent_names: List[str], executor: str, poses: Dict[str, airsim.Pose], target_mu: np.ndarray) -> Optional[str]:
    best = None
    for n in agent_names:
        if n == executor:
            continue
        d = _dist_xy(poses[n], target_mu)
        if (best is None) or (d < best[0]):
            best = (d, n)
    return best[1] if best else None

def MultiUAVCityNavAgent(
    scene_id: int,
    split: str,
    cfg: CoNavConfig,
    data_dir: str = "./data",
    max_step_size: int = 200,
    vlm_name: str = "sam",
    record: bool = False,
    only_episode_ids=None,
):
    """
    多无人机协同版本：
      - 每步仍复用单机的 observation->LLM->VLM->决策->控制链路
      - 任何无人机 LLM 认为 next landmark is_found 时：
          executor = 该无人机（冻结位置）
          生成点云+位置+语义置信度 -> post 到共享黑板
          分配 verifier 无人机飞到可观测点云的位置进行验证
      - executor&verifier 位置接近且置信度高 => confirm => landmark_idx++
      - 然后全队 regroup 到地标附近（移动同单机，途中可开启持续搜寻）
      - 失败则 verifier 回退到 executor 位置，继续探索
    """
    ENV_ID = scene_id  # 与原单机脚本一致的变量名约定

    data_root = os.path.join(data_dir, f"gt_by_env/{ENV_ID}/{split}_landmk.json")
    graph_root = os.path.join(data_dir, f"mem_graphs_pruned/{ENV_ID}/{split}")
    graph_act_root = os.path.join(data_dir, f'mem_graphs/{ENV_ID}.pkl')
    os.makedirs("obs_imgs", exist_ok=True)

    with open(data_root, 'r') as f:
        navi_tasks = json.load(f)['episodes']

    if only_episode_ids is not None:
        if isinstance(only_episode_ids, str):
            only_episode_ids = {only_episode_ids}
        else:
            only_episode_ids = set(only_episode_ids)
        navi_tasks = [e for e in navi_tasks if e.get('episode_id') in only_episode_ids]
        print(f"[Filter] Only running episodes: {[e['episode_id'] for e in navi_tasks]}")
        if not navi_tasks:
            raise RuntimeError("No matching episodes found for only_episode_ids")

    nav_evaluator = CityNavEvaluator()

    # load LLM (保持与你的单机一致；你可自行改 env var / key)
    llm = OpenAI_LLM_v2(
        max_tokens=10000,
        model_name="gpt-4o",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        client_type="openai",
        cache_name="navigation",
        finish_reasons=["stop", "length"],
    )

    # load VLM
    if vlm_name == "dino":
        vlm = load_model(
            "external/Grounded_Sam_Lite/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "external/Grounded_Sam_Lite/weights/groundingdino_swint_ogc.pth"
        )
    else:
        vlm = GroundedSam(
            dino_checkpoint_path="external/Grounded_Sam_Lite/weights/groundingdino_swint_ogc.pth",
            sam_checkpoint_path="external/Grounded_Sam_Lite/weights/sam_vit_h_4b8939.pth"
        )

    # load env
    machines_info_xxx = [
        {
            'MACHINE_IP': '127.0.0.1',
            'SOCKET_PORT': 30000,
            'MAX_SCENE_NUM': 8,
            'open_scenes': [scene_id],
        },
    ]
    tool = AirVLNSimulatorClientTool(machines_info=machines_info_xxx)
    run_ret = tool.run_call()

    # --- IMPORTANT: AirSim ApiServerPort is usually returned by AirVLN tool (e.g., 30001) ---
    tool_ports = _extract_airsim_ports_from_run_ret(run_ret)
    if tool_ports:
        print(f"[AirSim] ports from AirVLN tool: {tool_ports}")

    # raw AirSim client (prefer: --airsim_port > tool_ports > settings.json > fallback)
    raw_port = getattr(cfg, 'airsim_port', None)
    if raw_port is None and tool_ports:
        raw_port = tool_ports[0]
    raw_client = _make_raw_airsim_client(
        ip=machines_info_xxx[0].get('MACHINE_IP','127.0.0.1'),
        port=raw_port,
        port_candidates=tool_ports
    )
    # enable control for each UAV (safe if already enabled)
    for _vn in [f"UAV{i+1}" for i in range(cfg.num_uavs)]:
        try:
            raw_client.enableApiControl(True, vehicle_name=_vn)
            raw_client.armDisarm(True, vehicle_name=_vn)
        except Exception:
            pass

    agent_names = [f"UAV{i+1}" for i in range(cfg.num_uavs)]
    agent_tools = {n: AgentSpecificToolWrapper(tool, n, raw_client) for n in agent_names}

    # navigation pipeline
    for i in tqdm(range(len(navi_tasks))):
        navi_task = navi_tasks[i]
        episode_id = navi_task['episode_id']
        print(f"\n============================== Start Multi-UAV episode {episode_id} ==============================")

        mem_graph = NavigationGraph(os.path.join(graph_root, f"{episode_id}.pkl"))
        with open(graph_act_root, 'rb') as f:
            mem_act_graph = pickle.load(f)

        landmarks = navi_task["instruction"]["landmarks"]
        if len(landmarks) == 0:
            continue
        print("[Landmarks]", landmarks)
        instruction = navi_task["instruction"]['instruction_text']
        reference_path = navi_task['reference_path']

        # base pose from dataset
        base_pose = convert_airsim_pose(
            navi_task["start_position"] + navi_task["start_rotation"][1:] + [navi_task["start_rotation"][0]]
        )
        target_pose = convert_airsim_pose(navi_task["goals"][0]['position'] + [0, 0, 0, 1])

        # swarm init
        init_poses = _spawn_swarm_poses(base_pose, cfg)
        curr_poses: Dict[str, airsim.Pose] = {}
        for n, p in zip(agent_names, init_poses):
            agent_tools[n].setPoses([[p]])
            curr_poses[n] = p

        # takeoff (single-like)
        for n in agent_names:
            p = curr_poses[n]
            p2 = _takeoff_like_single(p, cfg.takeoff_steps)
            agent_tools[n].setPoses([[p2]])
            curr_poses[n] = p2

        # team state
        next_landmark_idx = 0
        scene_objects = []  # 保持与单机接口一致
        blackboard = TeamBlackboard()

        # bookkeeping
        step_budget = max_step_size
        step_used = {n: 0 for n in agent_names}
        sync_events = 0
        vlm_calls = 0

        # 记录轨迹（可扩展为每机一条）
        data_dict = {
            "episode_id": episode_id,
            "instruction": instruction,
            "gt_traj": [pose[:3] for pose in reference_path],
            "pred_traj": [],  # 用 UAV1 的轨迹兼容 evaluator
            "pred_traj_explore": [],
            "pred_traj_memory": [],
            "pred_traj_team": {n: [] for n in agent_names}
        }

        # helper: append pose to traj
        def _log_pose(n: str, pose: airsim.Pose):
            xyz = [pose.position.x_val, pose.position.y_val, pose.position.z_val]
            data_dict["pred_traj_team"][n].append(xyz)
            if n == agent_names[0]:
                data_dict["pred_traj"].append(xyz)

        for n in agent_names:
            _log_pose(n, curr_poses[n])

        # --- main loop ---
        global_iter = 0
        while global_iter < max_step_size and next_landmark_idx < len(landmarks):
            global_iter += 1
            lm_name = landmarks[next_landmark_idx]

            if cfg.verbose:
                print(f"\n[CO-UAV] iter={global_iter}, next_lm_idx={next_landmark_idx}, lm='{lm_name}', pending={blackboard.pending is not None}")

            # 1) 若黑板有 pending hypothesis -> 分配 verifier 进行验证
            if cfg.enable_verification and blackboard.pending is not None:
                hyp = blackboard.pending
                executor = hyp["executor"]
                verifier = hyp.get("verifier", None)
                if verifier is None:
                    verifier = _choose_verifier(agent_names, executor, curr_poses, hyp["pos_mu"])
                    hyp["verifier"] = verifier
                    print(f"[CO-UAV] assign verifier={verifier} for executor={executor}")

                if verifier is None:
                    print("[CO-UAV][WARN] no verifier available; fallback to single (auto-confirm)")
                    next_landmark_idx += 1
                    blackboard.clear()
                    continue

                # verifier executes verification
                verify_out = verifier_verify_hypothesis(
                    verifier_name=verifier,
                    verifier_tool=agent_tools[verifier],
                    verifier_pose=curr_poses[verifier],
                    executor_pose=curr_poses[executor],
                    hyp=hyp,
                    llm=llm, vlm=vlm,
                    scene_id=scene_id,
                    cfg=cfg
                )
                vlm_calls += 1  # rough accounting (实际是多次候选；这里给你留接口后续细化)

                curr_poses[verifier] = verify_out["view_pose"]
                _log_pose(verifier, curr_poses[verifier])

                # compare
                dist_ex_vf = float(np.linalg.norm(hyp["pos_mu"][:2] - verify_out["pos_mu"][:2]))
                conf_ex = float(hyp["sem_conf"])
                conf_vf = float(verify_out["sem_conf"])
                ok_dist = dist_ex_vf <= cfg.assoc_dist_m
                ok_conf = (conf_ex >= cfg.conf_accept) and (conf_vf >= cfg.conf_accept)
                llm_ok = bool(verify_out["llm_found"])

                print(f"[CO-UAV] VERIFY: dist={dist_ex_vf:.2f}m (<= {cfg.assoc_dist_m}), "
                      f"conf_ex={conf_ex:.2f}, conf_vf={conf_vf:.2f}, llm_found={llm_ok}")

                hyp["verify_out"] = verify_out
                hyp["dist_ex_vf"] = dist_ex_vf
                hyp["conf_vf"] = conf_vf
                hyp["llm_found_vf"] = llm_ok
                hyp["attempts"] = int(hyp.get("attempts", 0) + 1)

                # if ok_dist and ok_conf and llm_ok:
                if ok_dist and ok_conf:
                    print(f"[CO-UAV] ✅ VERIFIED landmark '{lm_name}' by {executor}+{verifier}. Confirm & regroup.")
                    sync_events += 1

                    # confirm pose: average of two means (or choose executor mean)
                    mu = 0.5 * hyp["pos_mu"] + 0.5 * verify_out["pos_mu"]
                    # face direction: keep executor yaw
                    ex_ori = curr_poses[executor].orientation
                    confirm_pose = airsim.Pose(airsim.Vector3r(float(mu[0]), float(mu[1]), float(mu[2])), ex_ori)

                    next_landmark_idx += 1
                    blackboard.clear()

                    # regroup
                    if cfg.enable_regroup:
                        # simple formation offsets to avoid exact overlap
                        for j, n in enumerate(agent_names):
                            off = (j - (cfg.num_uavs - 1) / 2.0) * 2.0
                            tgt = airsim.Pose(
                                airsim.Vector3r(confirm_pose.position.x_val, confirm_pose.position.y_val + off, confirm_pose.position.z_val),
                                confirm_pose.orientation
                            )
                            sz, mid_coords = calculate_movement_steps(curr_poses[n], tgt)
                            # "途中持续搜寻"（轻量实现：每隔几步，UAV1 peek 一次，并只入黑板不打断 regroup）
                            if cfg.enable_search_during_regroup and n == agent_names[0] and next_landmark_idx < len(landmarks):
                                stride = 5
                                for k in range(0, len(mid_coords), stride):
                                    # teleport to intermediate
                                    mid = mid_coords[k]
                                    mid_pose = convert_airsim_pose(mid[:3] + quat_to_list(curr_poses[n].orientation))
                                    agent_tools[n].setPoses([[mid_pose]])
                                    curr_poses[n] = mid_pose
                                    _log_pose(n, mid_pose)

                                    # peek next landmark (freeze if found: executor stays)
                                    pano_obs, pano_pose = get_pano_observations(curr_poses[n], agent_tools[n], scene_id=scene_id)
                                    pano_imgs = [pano_obs[6][0], pano_obs[7][0], pano_obs[0][0], pano_obs[1][0], pano_obs[2][0]]
                                    pano_deps = [pano_obs[6][1], pano_obs[7][1], pano_obs[0][1], pano_obs[1][1], pano_obs[2][1]]
                                    pano_poses = [pano_pose[6], pano_pose[7], pano_pose[0], pano_pose[1], pano_pose[2]]
                                    paths = []
                                    for vp, img in zip(["left","slightly_left","front","slightly_right","right"], pano_imgs):
                                        p = f"obs_imgs/{n}_regrouppeek_{vp}.png"
                                        cv2.imwrite(p, img)
                                        paths.append(p)
                                    sz2, new_pose2, found2, obs_mu2, cov2, geom2, pc_bundle2 = explore_pipeline_by_sam(
                                        curr_poses[n], llm, vlm,
                                        cfg,
                                        paths, pano_imgs, pano_deps, pano_poses,
                                        instruction, scene_objects, landmarks, next_landmark_idx,
                                        freeze_on_found=True
                                    )
                                    if found2 and cfg.enable_blackboard and blackboard.pending is None:
                                        npts2 = int(pc_bundle2.get("npts", 0))
                                        vlm_raw2 = float(pc_bundle2.get("vlm_raw", 0.0))
                                        sem_conf2 = combine_semantic_conf(True, vlm_raw2, cfg, npts=npts2)
                                        if (npts2 < cfg.min_hyp_points) or (vlm_raw2 < cfg.min_vlm_raw_for_hyp):
                                            print(f"[CO-UAV][peek] (skip hyp) insufficient measurement: npts={npts2} (<{cfg.min_hyp_points}) "
                                                  f"or vlm_raw={vlm_raw2:.3f} (<{cfg.min_vlm_raw_for_hyp}).")
                                            continue
                                        hyp2 = {
                                            "landmark_idx": next_landmark_idx,
                                            "landmark_name": landmarks[next_landmark_idx],
                                            "executor": n,
                                            "pos_mu": obs_mu2.astype(np.float32),
                                            "cov_xy": cov2.astype(np.float32),
                                            "geom_conf": float(geom2),
                                            "vlm_raw": float(pc_bundle2.get("vlm_raw", 0.0)),
                                            "sem_conf": float(sem_conf2),
                                            "pc": pc_bundle2.get("pc", np.zeros((1,3), dtype=np.float32)),
                                            "npts": int(pc_bundle2.get("npts", 0)),
                                            "attempts": 0,
                                        }
                                        print(f"[CO-UAV][peek] queued next landmark '{hyp2['landmark_name']}' from {n}, conf={sem_conf2:.2f}")
                                        blackboard.post(hyp2)
                                        break

                            # final teleport to target
                            agent_tools[n].setPoses([[tgt]])
                            curr_poses[n] = tgt
                            _log_pose(n, tgt)
                            step_used[n] += int(sz)

                    continue  # go next iter

                # failed -> backtrack
                if hyp["attempts"] >= cfg.max_verify_attempts:
                    print(f"[CO-UAV] ❌ VERIFY FAILED (attempts={hyp['attempts']}). Backtrack verifier to executor and resume search.")
                    # backtrack verifier to executor pose
                    back_pose = curr_poses[executor]
                    agent_tools[verifier].setPoses([[back_pose]])
                    curr_poses[verifier] = back_pose
                    _log_pose(verifier, back_pose)
                    blackboard.clear()
                    continue
                else:
                    print(f"[CO-UAV] retry verification later (attempts={hyp['attempts']})")
                    continue

            # 2) 否则：各无人机复用单机 explore 流程并行(串行调用)探索 next landmark
            found_executor = None
            found_hyp = None
            memory_exploit_triggered = False
            memory_exploit_by = None

            for n in agent_names:
                # get pano obs
                pano_obs, pano_pose = get_pano_observations(curr_poses[n], agent_tools[n], scene_id=scene_id)
                pano_imgs = [pano_obs[6][0], pano_obs[7][0], pano_obs[0][0], pano_obs[1][0], pano_obs[2][0]]
                pano_deps = [pano_obs[6][1], pano_obs[7][1], pano_obs[0][1], pano_obs[1][1], pano_obs[2][1]]
                pano_poses = [pano_pose[6], pano_pose[7], pano_pose[0], pano_pose[1], pano_pose[2]]

                paths = []
                for vp, img in zip(["left","slightly_left","front","slightly_right","right"], pano_imgs):
                    p = f"obs_imgs/{n}_rgb_{vp}.png"
                    cv2.imwrite(p, img)
                    paths.append(p)

                # === memory-graph exploit (ported from SimRun single) ===
                # 为避免多机团队状态/黑板语义冲突，这里先在评估主轨迹 UAV1 上启用；
                # 一旦触发，直接走记忆图规划并结束当前 episode（与单机逻辑一致）。
                if n == agent_names[0]:
                    cls_node = find_closest_node(mem_graph._graph, list(curr_poses[n].position), thresh=20)

                    if cls_node is not None:
                        print(f"[CO-UAV][{n}] Find the memory graph node!!!")

                        # 使用 front + back 两张图构造观测（与 SimRun 更一致）
                        try:
                            front_img = pano_obs[0][0]
                            back_img = pano_obs[4][0]
                            okf, encf = cv2.imencode(".png", front_img)
                            okb, encb = cv2.imencode(".png", back_img)
                            obs_imgs_bytes = []
                            if okf:
                                obs_imgs_bytes.append(encf.tobytes())
                            if okb:
                                obs_imgs_bytes.append(encb.tobytes())
                        except Exception:
                            obs_imgs_bytes = []

                        obs = {
                            "pos": np.array(list(curr_poses[n].position)),
                            "image": obs_imgs_bytes,
                        }
                        new_node = mem_graph.add_vertix(obs)
                        mem_graph.add_edge(new_node, cls_node)

                        rest_landmarks = landmarks[next_landmark_idx:]
                        result = pipeline.full_pipeline(mem_graph, start_node=new_node, landmarks=rest_landmarks, alpha=0.0001)

                        walk = [a[0] for a in result["walk"]]
                        node_traj = [mem_graph.get_node_data(node)["position"].tolist() for node in walk]
                        sz_mem, action_traj = calculate_movement_steps_mem(mem_act_graph, node_traj)

                        rest_steps = int(min(step_budget - max(step_used.values()), sz_mem))
                        rest_walks = action_traj[:rest_steps]

                        if len(rest_walks) > 0:
                            xyz_walks = [w[:3] for w in rest_walks]
                            data_dict["pred_traj_memory"].extend(rest_walks)
                            data_dict["pred_traj_team"][n].extend(xyz_walks)
                            # evaluator 仍使用 UAV1 作为 pred_traj
                            data_dict["pred_traj"].extend(xyz_walks)

                            stop_pos = rest_walks[-1][:3]
                            curr_poses[n] = convert_airsim_pose(list(stop_pos) + quat_to_list(curr_poses[n].orientation))
                            agent_tools[n].setPoses([[curr_poses[n]]])

                            step_used[n] += int(rest_steps)
                            memory_exploit_triggered = True
                            memory_exploit_by = n
                            print(f"[CO-UAV][{n}] memory exploit walk_steps={rest_steps}, episode will end with memory route.")
                            break
                        else:
                            print(f"[CO-UAV][{n}] memory node found but no executable memory walk (rest_steps={rest_steps}). Continue explore.")

                sz, new_pose, found, obs_mu, cov_xy, geom_conf, pc_bundle = explore_pipeline_by_sam(
                    curr_poses[n], llm, vlm,
                    cfg,
                    paths, pano_imgs, pano_deps, pano_poses,
                    instruction, scene_objects, landmarks, next_landmark_idx,
                    freeze_on_found=cfg.freeze_executor_on_found
                )

                if found:
                    # If LLM says found but VLM measurement is weak/empty, keep exploring (do NOT freeze)
                    npts_tmp = int(pc_bundle.get("npts", 0))
                    vlm_raw_tmp = float(pc_bundle.get("vlm_raw", 0.0))
                    if cfg.require_measurement_for_found and ((npts_tmp < cfg.min_hyp_points) or (vlm_raw_tmp < cfg.min_vlm_raw_for_hyp)):
                        found = False
                    else:
                        # executor should not move
                        new_pose = curr_poses[n] if cfg.freeze_executor_on_found else new_pose
                        sz = 0

                # move
                agent_tools[n].setPoses([[new_pose]])
                curr_poses[n] = new_pose
                _log_pose(n, new_pose)

                step_used[n] += int(sz)

                # if found -> propose hypothesis to blackboard
                if found and cfg.enable_blackboard and blackboard.pending is None:
                    npts = int(pc_bundle.get("npts", 0))
                    vlm_raw = float(pc_bundle.get("vlm_raw", 0.0))
                    sem_conf = combine_semantic_conf(True, vlm_raw, cfg, npts=npts)
                    print(f"[CO-UAV][{n}] FOUND '{lm_name}'. vlm_raw={vlm_raw:.3f}, sem_conf={sem_conf:.2f}, npts={npts}")

                    # gate: avoid posting empty/noisy hypotheses (prevents endless verify loops)
                    if (npts < cfg.min_hyp_points) or (vlm_raw < cfg.min_vlm_raw_for_hyp):
                        print(f"[CO-UAV][{n}] (skip hyp) insufficient measurement: npts={npts} (<{cfg.min_hyp_points}) "
                              f"or vlm_raw={vlm_raw:.3f} (<{cfg.min_vlm_raw_for_hyp}). Continue exploring.")
                        continue

                    hyp = {
                        "landmark_idx": next_landmark_idx,
                        "landmark_name": lm_name,
                        "executor": n,
                        "pos_mu": obs_mu.astype(np.float32),
                        "cov_xy": cov_xy.astype(np.float32),
                        "geom_conf": float(geom_conf),
                        "vlm_raw": float(vlm_raw),
                        "sem_conf": float(sem_conf),
                        "pc": pc_bundle.get("pc", np.zeros((1, 3), dtype=np.float32)),
                        "npts": int(npts),
                        "attempts": 0,
                    }
                    blackboard.post(hyp)
                    found_executor = n
                    found_hyp = hyp
                    break

                # simple termination condition to avoid runaway
                if max(step_used.values()) >= step_budget:
                    break

            if memory_exploit_triggered:
                print(f"[CO-UAV] Episode terminated by memory-graph exploit from {memory_exploit_by}.")
                break

            if max(step_used.values()) >= step_budget:
                print("[CO-UAV] Step budget reached. End episode.")
                break

            # ablation: no verification -> directly increment on found
            if (not cfg.enable_verification) and (found_executor is not None):
                print(f"[CO-UAV][Ablation] verification disabled. Auto-confirm landmark '{lm_name}'.")
                next_landmark_idx += 1
                blackboard.clear()

        # episode done -> evaluate by UAV1 final pose
        stop_pose = curr_poses[agent_names[0]]
        stop_pos = np.array([stop_pose.position.x_val, stop_pose.position.y_val, stop_pose.position.z_val], dtype=np.float32)
        tgt_pos = np.array([target_pose.position.x_val, target_pose.position.y_val, target_pose.position.z_val], dtype=np.float32)
        ne = float(np.linalg.norm(tgt_pos - stop_pos))
        
        gt_goal = np.array(reference_path[-1][:3], dtype=np.float32)  # 和 evaluator 的 gt_traj[-1]一致
        ne = float(np.linalg.norm(gt_goal - stop_pos))


        success = ne < 20.0
        data_dict.update({"success": success})
        print(f"\n############## Multi-UAV Episode {episode_id}: {'success' if success else 'failed'}  NE={ne:.2f}")

        nav_evaluator.update(data_dict)
        nav_evaluator.log_metrics()

    nav_evaluator.log_metrics()

# ==================== 8) 仍保留单机入口（基本不改） ====================

def CityNavAgent(scene_id, split, data_dir="./data", max_step_size=200, vlm_name="dino", record=False,
                 only_episode_ids=None):

    ENV_ID = scene_id

    data_root = os.path.join(data_dir, f"gt_by_env/{ENV_ID}/{split}_landmk.json")
    graph_root = os.path.join(data_dir, f"mem_graphs_pruned/{ENV_ID}/{split}")
    graph_act_root = os.path.join(data_dir, f'mem_graphs/{ENV_ID}.pkl')

    os.makedirs("obs_imgs", exist_ok=True)

    predict_routes = []
    with open(data_root, 'r') as f:
        navi_tasks = json.load(f)['episodes']

    # ====== 新增：白名单过滤 ======
    if only_episode_ids is not None:
        if isinstance(only_episode_ids, str):
            only_episode_ids = {only_episode_ids}
        else:
            only_episode_ids = set(only_episode_ids)
        navi_tasks = [e for e in navi_tasks if e.get('episode_id') in only_episode_ids]
        print(f"[Filter] Only running episodes: {[e['episode_id'] for e in navi_tasks]}")
        if not navi_tasks:
            raise RuntimeError("No matching episodes found for only_episode_ids")

    nav_evaluator = CityNavEvaluator()

    # load LLM
    llm = OpenAI_LLM_v2(
        max_tokens=10000,
        model_name="gpt-4o",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        client_type="openai",
        cache_name="navigation",
        finish_reasons=["stop", "length"],
    )

    if vlm_name == "dino":
        vlm = load_model(
            "external/Grounded_Sam_Lite/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "external/Grounded_Sam_Lite/weights/groundingdino_swint_ogc.pth"
        )
    else:
        vlm = GroundedSam(
            dino_checkpoint_path="external/Grounded_Sam_Lite/weights/groundingdino_swint_ogc.pth",
            sam_checkpoint_path="external/Grounded_Sam_Lite/weights/sam_vit_h_4b8939.pth"
        )

    # load env
    machines_info_xxx = [
        {
            'MACHINE_IP': '127.0.0.1',
            'SOCKET_PORT': 30000,
            'MAX_SCENE_NUM': 8,
            'open_scenes': [scene_id],
        },
    ]

    tool = AirVLNSimulatorClientTool(machines_info=machines_info_xxx)
    tool.run_call()

    # single-mode config for shared helper functions
    cfg_single = CoNavConfig(num_uavs=1, enable_blackboard=False, enable_verification=False, enable_regroup=False, verbose=False)


    # navigation pipeline
    for i in tqdm(range(len(navi_tasks))):
        navi_task = navi_tasks[i]
        episode_id = navi_task['episode_id']

        print(f"================================ Start episode {episode_id} ==================================")
        mem_graph = NavigationGraph(os.path.join(graph_root, f"{episode_id}.pkl"))
        with open(graph_act_root, 'rb') as f:
            mem_act_graph = pickle.load(f)

        landmarks = navi_task["instruction"]["landmarks"]
        if len(landmarks) == 0:
            continue

        next_landmark_idx = 0
        object_info = []
        instruction = navi_task["instruction"]['instruction_text']
        reference_path = navi_task['reference_path']

        step_size = 0
        hist_step_size = []

        curr_pose = convert_airsim_pose(navi_task["start_position"]+navi_task["start_rotation"][1:]+[navi_task["start_rotation"][0]])
        target_pose = convert_airsim_pose(navi_task["goals"][0]['position']+[0, 0, 0, 1])

        # set env
        tool.setPoses([[curr_pose]])

        # take off
        for _ in range(5):
            new_pose = getPoseAfterMakeActions(curr_pose, [4])
            curr_pose = new_pose
            tool.setPoses([[curr_pose]])

        while step_size < max_step_size:

            # get observation
            try:
                pano_obs, pano_pose = get_pano_observations(curr_pose, tool, scene_id=scene_id)
                pano_obs_imgs = [pano_obs[6][0], pano_obs[7][0], pano_obs[0][0], pano_obs[1][0], pano_obs[2][0]]
                pano_obs_deps = [pano_obs[6][1], pano_obs[7][1], pano_obs[0][1], pano_obs[1][1], pano_obs[2][1]]
                pano_obs_poses = [pano_pose[6], pano_pose[7], pano_pose[0], pano_pose[1], pano_pose[2]]

                pano_obs_imgs_path = ["obs_imgs/rgb_obs_{}.png".format(view_drc.replace(" ", "_")) for view_drc in
                                      ["left","slightly_left","front","slightly_right","right"]]
                for j in range(len(pano_obs_imgs_path)):
                    cv2.imwrite(pano_obs_imgs_path[j], pano_obs_imgs[j])

            except Exception as e:
                print(f"Task idx: {i}. Step size: {step_size}. Success: False. Failed to get images. Exception: {e}")
                break

            cls_node = find_closest_node(mem_graph._graph, list(curr_pose.position), thresh=20)

            # exploit
            if cls_node is not None:
                print("Find the memory graph node!!!")
                with open(pano_obs_imgs_path[2], "rb") as file:
                    imgf = file.read()
                obs = {
                    "pos": np.array(list(curr_pose.position)),
                    "image": [imgf]
                }
                new_node = mem_graph.add_vertix(obs)
                mem_graph.add_edge(new_node, cls_node)

                rest_landmarks = landmarks[next_landmark_idx:]
                result = pipeline.full_pipeline(mem_graph, start_node=new_node, landmarks=rest_landmarks, alpha=0.0001)

                walk = [a[0] for a in result["walk"]]
                node_traj = [mem_graph.get_node_data(node)["position"].tolist() for node in walk]
                sz, action_traj = calculate_movement_steps_mem(mem_act_graph, node_traj)

                rest_steps = int(min(max_step_size-step_size, sz))
                rest_walks = action_traj[:rest_steps]

                stop_pos = rest_walks[-1][:3]
                curr_pose = convert_airsim_pose(list(stop_pos) + quat_to_list(curr_pose.orientation))
                tool.setPoses([[curr_pose]])

                step_size += rest_steps
                break

            # explore
            else:
                print("No memory graph reached, keep exploring ...")

                sz, new_pose, next_landmark_found, *_ = explore_pipeline_by_sam(
                    curr_pose, llm, vlm,
                    cfg_single,
                    pano_obs_imgs_path,
                    pano_obs_imgs,
                    pano_obs_deps,
                    pano_obs_poses,
                    instruction, object_info, landmarks, next_landmark_idx,
                    freeze_on_found=False
                )

                tool.setPoses([[new_pose]])
                curr_pose = new_pose
                step_size += sz
                hist_step_size.append(sz)

                if next_landmark_found:
                    next_landmark_idx += 1

                if next_landmark_idx >= len(landmarks):
                    print(f"Task idx: {i}. Total steps: {step_size}. Exploration finished.")
                    break

                if len(hist_step_size) >= 4 and sum(hist_step_size[-4:-1]) == 0.0:
                    print(f"Task idx: {i}. Total steps: {step_size}. Success: False. Stuck!!")
                    break

        stop_pos = np.array(list(curr_pose.position))
        target_pos = np.array(list(target_pose.position))
        ne = np.linalg.norm(np.array(target_pos) - np.array(stop_pos))

        if ne < 20:
            print(f"############## Episode {episode_id}: success, NE: {ne}. Step size: {step_size}")
        else:
            print(f"############## Episode {episode_id}: failed. NE: {ne}")

    nav_evaluator.log_metrics()

# ==================== 9) CLI ====================

def _build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="multi", choices=["single","multi"])
    p.add_argument("--env_id", type=int, default=3)
    p.add_argument("--split", type=str, default="val_seen")
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument("--vlm", type=str, default="sam", choices=["sam","dino"])
    p.add_argument("--only_episode", type=str, default=None)
    # multi-uav
    p.add_argument("--num_uavs", type=int, default=2)
    p.add_argument("--airsim_port", type=int, default=None, help="AirSim ApiServerPort; if None read settings.json or fallback 41451")
    p.add_argument("--spawn_sep", type=float, default=5.0)
    p.add_argument("--spawn_axis", type=str, default="z", choices=["y","z"])
    p.add_argument("--no_verify", action="store_true")
    p.add_argument("--no_regroup", action="store_true")
    p.add_argument("--no_blackboard", action="store_true")
    p.add_argument("--no_regroup_search", action="store_true")
    p.add_argument("--assoc_dist", type=float, default=20.0)
    p.add_argument("--conf_accept", type=float, default=0.50)
    p.add_argument("--min_hyp_points", type=int, default=120)
    p.add_argument("--min_vlm_raw_for_hyp", type=float, default=0.2)
    p.add_argument("--no_require_meas", action="store_true", help="Allow LLM-found without VLM measurement (not recommended)")
    p.add_argument("--no_fallback_motion", action="store_true", help="Disable fallback motion when no measurement")
    p.add_argument("--fallback_step", type=float, default=6.0)
    p.add_argument("--smooth_related_move", action="store_false", help="Ablation: use smoothed move toward proxy target instead of direct fly")
    p.add_argument("--sem_w_vlm", type=float, default=0.8)
    p.add_argument("--sem_w_llm", type=float, default=0.2)
    p.add_argument("--verbose", action="store_true")
    return p

if __name__ == '__main__':
    args = _build_argparser().parse_args()
    # only_episode="3VELCLL3GTHB2UJSCD7IC4U05DK1F0" 
    only_episode=None 
    if args.mode == "single":
        CityNavAgent(args.env_id, args.split, max_step_size=args.max_steps, vlm_name=args.vlm, record=False, only_episode_ids=only_episode)
    else:
        cfg = CoNavConfig(
            num_uavs=args.num_uavs,
            airsim_port=args.airsim_port,
            spawn_sep_m=args.spawn_sep,
            spawn_axis=args.spawn_axis,
            enable_verification=(not args.no_verify),
            enable_regroup=(not args.no_regroup),
            enable_blackboard=(not args.no_blackboard),
            enable_search_during_regroup=(not args.no_regroup_search),
            assoc_dist_m=args.assoc_dist,
            conf_accept=args.conf_accept,
            min_hyp_points=args.min_hyp_points,
            min_vlm_raw_for_hyp=args.min_vlm_raw_for_hyp,
            require_measurement_for_found=(not args.no_require_meas),
            enable_fallback_motion=(not args.no_fallback_motion),
            fallback_step_m=args.fallback_step,
            direct_fly_to_related=(not args.smooth_related_move),
            sem_w_vlm=args.sem_w_vlm,
            sem_w_llm=args.sem_w_llm,
            verbose=args.verbose or True
        )
        MultiUAVCityNavAgent(args.env_id, args.split, cfg, max_step_size=args.max_steps, vlm_name=args.vlm, record=False, only_episode_ids=only_episode)
