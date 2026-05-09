import argparse
import json
import math
import os
import random
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import airsim
import cv2
import matplotlib
import msgpackrpc
import numpy as np
from scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.maps import get_IntrinsicMatrix, build_local_point_cloud, build_global_point_cloud


def _read_airsim_port_from_settings(default_port: int = 30001) -> int:
    candidates: List[str] = []
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "settings.json"))
    candidates.append(os.path.join(os.getcwd(), "settings.json"))
    home = os.path.expanduser("~")
    candidates.append(os.path.join(home, "Documents", "AirSim", "settings.json"))
    candidates.append(os.path.join(home, ".config", "AirSim", "settings.json"))
    candidates.append(os.path.join(home, ".airsim", "settings.json"))
    env_p = os.environ.get("AIRSIM_SETTINGS_PATH", "")
    if env_p:
        candidates.append(env_p)

    for p in candidates:
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    s = json.load(f)
                port_v = s.get("ApiServerPort", default_port)
                if port_v is None:
                    continue
                return int(port_v)
        except Exception:
            continue
    return int(default_port)


def _make_client(ip: str, port: Optional[int]) -> Tuple[airsim.MultirotorClient, int]:
    candidates: List[int] = []
    if port is not None:
        candidates.append(int(port))
    candidates.extend([
        _read_airsim_port_from_settings(default_port=30001),
        30001,
        41451,
    ])
    uniq = []
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        uniq.append(int(p))
        seen.add(int(p))

    last_err = None
    for p in uniq:
        try:
            c = airsim.MultirotorClient(ip=ip, port=int(p))
            c.confirmConnection()
            print(f"[AirSim] connected to {ip}:{p}")
            return c, int(p)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Cannot connect to AirSim on {ip}. Tried ports={uniq}. "
        f"Please start scene/server first. Last error: {last_err}"
    )


def _extract_airsim_ports_from_server_ret(run_ret) -> List[int]:
    ports: List[int] = []

    def _walk(x):
        if x is None:
            return
        # Pattern: [ip_str, [port_int, ...]]
        if isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], str) and isinstance(x[1], (list, tuple)):
            if all(isinstance(p, int) for p in x[1]):
                ports.extend([int(p) for p in x[1]])
                return
        if isinstance(x, dict):
            for v in x.values():
                _walk(v)
        elif isinstance(x, (list, tuple)):
            for it in x:
                _walk(it)

    _walk(run_ret)
    uniq: List[int] = []
    seen = set()
    for p in ports:
        if p not in seen:
            uniq.append(int(p))
            seen.add(int(p))
    return uniq


def _reopen_scene_via_server_tool(
    host: str,
    simulator_tool_port: int,
    scene_id: int,
    timeout_sec: int = 180,
) -> Optional[int]:
    """
    Ask AirVLNSimulatorServerTool(30000) to open the scene, then return AirSim ApiServerPort.
    """
    client = msgpackrpc.Client(msgpackrpc.Address(host, int(simulator_tool_port)), timeout=timeout_sec)
    try:
        _ = client.call("ping")
        ret = client.call("reopen_scenes", host, [int(scene_id)])
    finally:
        try:
            client.close()
        except Exception:
            pass

    # Expected ret like: [True, ['127.0.0.1', [30001]]]
    if not isinstance(ret, (list, tuple)) or len(ret) < 2:
        return None
    if ret[0] is not True:
        return None
    ports = _extract_airsim_ports_from_server_ret(ret[1])
    if len(ports) == 0:
        ports = _extract_airsim_ports_from_server_ret(ret)
    if len(ports) == 0:
        return None
    return int(ports[0])


def _resolve_vehicle_name(client: airsim.MultirotorClient, preferred: str, fallback: str = "Drone_1") -> str:
    for n in [preferred, fallback]:
        try:
            _ = client.simGetVehiclePose(vehicle_name=n)
            return n
        except Exception:
            continue
    return preferred


def _read_image_shape_from_param_py(param_py_path: str) -> Dict[str, int]:
    defaults = {
        "Image_Height_RGB": 512,
        "Image_Width_RGB": 512,
        "Image_Height_DEPTH": 512,
        "Image_Width_DEPTH": 512,
    }
    if not os.path.exists(param_py_path):
        return defaults

    try:
        with open(param_py_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return defaults

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for k in list(defaults.keys()):
            m = re.search(rf"--{k}'?\s*,\s*type=int,\s*default=(\d+)", line)
            if m:
                defaults[k] = int(m.group(1))
    return defaults


def _pose_to_list(p: airsim.Pose) -> List[float]:
    return [
        float(p.position.x_val),
        float(p.position.y_val),
        float(p.position.z_val),
        float(p.orientation.x_val),
        float(p.orientation.y_val),
        float(p.orientation.z_val),
        float(p.orientation.w_val),
    ]


def _set_vehicle_pose_xyz_yaw(
    client: airsim.MultirotorClient,
    x: float,
    y: float,
    z: float,
    yaw_deg: float,
    vehicle_name: str,
) -> airsim.Pose:
    yaw = math.radians(float(yaw_deg))
    q = R.from_euler("z", yaw, degrees=False).as_quat()
    p = airsim.Pose(
        position_val=airsim.Vector3r(float(x), float(y), float(z)),
        orientation_val=airsim.Quaternionr(x_val=float(q[0]), y_val=float(q[1]), z_val=float(q[2]), w_val=float(q[3])),
    )
    client.simSetVehiclePose(p, True, vehicle_name=vehicle_name)
    return p


def _get_camera_pose(client: airsim.MultirotorClient, camera_id: str, vehicle_name: str) -> airsim.Pose:
    try:
        info = client.simGetCameraInfo(camera_id, vehicle_name=vehicle_name)
        return info.pose
    except Exception:
        return client.simGetVehiclePose(vehicle_name=vehicle_name)


def _get_camera_fov_deg(client: airsim.MultirotorClient, camera_id: str, vehicle_name: str, fallback: float = 90.0) -> float:
    try:
        info = client.simGetCameraInfo(camera_id, vehicle_name=vehicle_name)
        fov = float(getattr(info, "fov", fallback))
        if fov > 1e-3:
            return fov
    except Exception:
        pass
    return float(fallback)


def _capture_rgb_depth_seg(
    client: airsim.MultirotorClient,
    vehicle_name: str,
    camera_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    req = [
        airsim.ImageRequest(camera_id, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(camera_id, airsim.ImageType.DepthPerspective, True, False),
        airsim.ImageRequest(camera_id, airsim.ImageType.Segmentation, False, False),
    ]
    resp = client.simGetImages(req, vehicle_name=vehicle_name)
    if len(resp) != 3:
        raise RuntimeError(f"simGetImages expects 3 responses, got {len(resp)}")

    h = resp[0].height
    w = resp[0].width
    rgb = np.frombuffer(resp[0].image_data_uint8, dtype=np.uint8).reshape(h, w, 3)
    depth = np.array(resp[1].image_data_float, dtype=np.float32).reshape(resp[1].height, resp[1].width)
    seg = np.frombuffer(resp[2].image_data_uint8, dtype=np.uint8).reshape(resp[2].height, resp[2].width, 3)
    return rgb, depth, seg


def _get_lidar_data(client: airsim.MultirotorClient, vehicle_name: str, lidar_name: str):
    try:
        return client.getLidarData(lidar_name=lidar_name, vehicle_name=vehicle_name)
    except Exception:
        try:
            return client.simGetLidarData(lidar_name=lidar_name, vehicle_name=vehicle_name)
        except Exception:
            return None


def _lidar_points_to_depth(
    lidar_points_xyz: np.ndarray,
    height: int,
    width: int,
    fov_deg: float,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    depth = np.zeros((height, width), dtype=np.float32)
    if lidar_points_xyz.size == 0:
        return depth

    x = lidar_points_xyz[:, 0]  # forward
    y = lidar_points_xyz[:, 1]  # right
    z = lidar_points_xyz[:, 2]  # down

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (x > float(min_depth)) & (x < float(max_depth))
    if not np.any(valid):
        return depth

    x = x[valid]
    y = y[valid]
    z = z[valid]

    fx = width / (2.0 * np.tan(np.deg2rad(fov_deg / 2.0)))
    fy = height / (2.0 * np.tan(np.deg2rad(fov_deg / 2.0)))
    cx = width / 2.0
    cy = height / 2.0

    u = np.round(fx * (y / x) + cx).astype(np.int32)
    v = np.round(fy * (z / x) + cy).astype(np.int32)
    in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(in_img):
        return depth

    u = u[in_img]
    v = v[in_img]
    x = x[in_img]

    # z-buffer by nearest x (forward depth)
    for px, py, d in zip(u, v, x):
        old = depth[py, px]
        if old <= 0.0 or d < old:
            depth[py, px] = float(d)
    return depth


def _capture_rgb_seg_with_lidar_depth(
    client: airsim.MultirotorClient,
    vehicle_name: str,
    camera_id: str,
    lidar_vehicle_name: str,
    lidar_name: str,
    fov_deg: float,
    min_depth: float,
    max_depth: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    rgb, depth_backup, seg = _capture_rgb_depth_seg(client, vehicle_name=vehicle_name, camera_id=camera_id)

    lidar_meta: Dict[str, Any] = {
        "depth_source": "depth_perspective_fallback",
        "lidar_vehicle": lidar_vehicle_name,
        "lidar_name": lidar_name,
        "lidar_points_raw": 0,
        "lidar_points_used": 0,
    }

    lidar_data = _get_lidar_data(client, vehicle_name=lidar_vehicle_name, lidar_name=lidar_name)
    if lidar_data is None or len(getattr(lidar_data, "point_cloud", [])) < 3:
        return rgb, depth_backup, seg, lidar_meta

    lidar_pts = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    lidar_meta["lidar_points_raw"] = int(lidar_pts.shape[0])

    h, w = depth_backup.shape[:2]
    depth_lidar = _lidar_points_to_depth(
        lidar_points_xyz=lidar_pts,
        height=h,
        width=w,
        fov_deg=fov_deg,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    used = int(np.count_nonzero(depth_lidar > 0))
    lidar_meta["lidar_points_used"] = used

    # Sparse LiDAR can be empty in some frames; fallback to image depth.
    if used > 200:
        lidar_meta["depth_source"] = "lidar_raytrace_projection"
        return rgb, depth_lidar.astype(np.float32), seg, lidar_meta
    return rgb, depth_backup, seg, lidar_meta


def _center_crop_resize(arr: np.ndarray, target_h: int, target_w: int, is_label: bool = False) -> np.ndarray:
    src_h, src_w = arr.shape[:2]
    if src_h == target_h and src_w == target_w:
        return arr

    src_ratio = float(src_w) / float(src_h)
    tgt_ratio = float(target_w) / float(target_h)

    if src_ratio > tgt_ratio:
        # source too wide -> crop width
        crop_h = src_h
        crop_w = int(round(crop_h * tgt_ratio))
    else:
        # source too tall -> crop height
        crop_w = src_w
        crop_h = int(round(crop_w / tgt_ratio))

    crop_w = max(1, min(crop_w, src_w))
    crop_h = max(1, min(crop_h, src_h))
    x0 = max(0, (src_w - crop_w) // 2)
    y0 = max(0, (src_h - crop_h) // 2)

    if arr.ndim == 2:
        crop = arr[y0:y0 + crop_h, x0:x0 + crop_w]
    else:
        crop = arr[y0:y0 + crop_h, x0:x0 + crop_w, :]

    interp = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
    return cv2.resize(crop, (target_w, target_h), interpolation=interp)


def _align_modalities(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    seg: np.ndarray,
    target_h: int,
    target_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, List[int]]]:
    rgb_aligned = _center_crop_resize(rgb, target_h=target_h, target_w=target_w, is_label=False)
    depth_aligned = _center_crop_resize(depth_m, target_h=target_h, target_w=target_w, is_label=False)
    seg_aligned = _center_crop_resize(seg, target_h=target_h, target_w=target_w, is_label=True)
    meta = {
        "raw_rgb_hw": [int(rgb.shape[0]), int(rgb.shape[1])],
        "raw_depth_hw": [int(depth_m.shape[0]), int(depth_m.shape[1])],
        "raw_seg_hw": [int(seg.shape[0]), int(seg.shape[1])],
        "aligned_hw": [int(target_h), int(target_w)],
    }
    return rgb_aligned, depth_aligned, seg_aligned, meta


def _depth_to_global_pc(
    depth_m: np.ndarray,
    camera_pose: airsim.Pose,
    fov_deg: float = 90.0,
) -> np.ndarray:
    h, w = depth_m.shape[:2]
    K = get_IntrinsicMatrix(fov=fov_deg, height=h, width=w)
    local_pc = build_local_point_cloud(depth_m, K)
    cam_pose_arr = np.array(_pose_to_list(camera_pose), dtype=np.float32)
    world_pc = build_global_point_cloud(local_pc, cam_pose_arr)
    return world_pc


def _semantic_point_cloud(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    seg: np.ndarray,
    camera_pose: airsim.Pose,
    fov_deg: float,
    min_depth: float,
    max_depth: float,
) -> Dict[str, np.ndarray]:
    # Require already aligned arrays to guarantee point-image one-to-one mapping.
    h, w = depth_m.shape[:2]
    if rgb.shape[:2] != (h, w) or seg.shape[:2] != (h, w):
        raise ValueError(
            f"Modality shape mismatch after alignment: "
            f"rgb={rgb.shape[:2]}, depth={depth_m.shape[:2]}, seg={seg.shape[:2]}"
        )

    world_pc = _depth_to_global_pc(depth_m, camera_pose, fov_deg=fov_deg)
    valid = (depth_m > float(min_depth)) & (depth_m < float(max_depth)) & np.isfinite(depth_m)

    pts = world_pc[valid].reshape(-1, 3).astype(np.float32)
    rgb_pts = rgb[valid].reshape(-1, 3).astype(np.uint8)
    sem_rgb = seg[valid].reshape(-1, 3).astype(np.uint8)
    sem_ids = (
        sem_rgb[:, 0].astype(np.uint32) * 65536
        + sem_rgb[:, 1].astype(np.uint32) * 256
        + sem_rgb[:, 2].astype(np.uint32)
    ).astype(np.uint32)

    return {
        "points_xyz": pts,
        "points_rgb": rgb_pts,
        "points_sem_rgb": sem_rgb,
        "points_sem_id": sem_ids,
    }


def _write_ascii_ply_semantic(path: str, points_xyz: np.ndarray, sem_rgb: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = int(points_xyz.shape[0])
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points_xyz[i]
            r, g, b = sem_rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def _save_visualization(
    out_png: str,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    seg: np.ndarray,
    points_xyz: np.ndarray,
    sem_rgb: np.ndarray,
    weather_name: str,
    sample_id: int,
):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    if points_xyz.shape[0] > 50000:
        idx = np.random.choice(points_xyz.shape[0], 50000, replace=False)
        pc = points_xyz[idx]
        c = sem_rgb[idx]
    else:
        pc = points_xyz
        c = sem_rgb

    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    ax1.set_title("RGB")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    d = np.clip(depth_m, 0.0, np.percentile(depth_m[np.isfinite(depth_m)], 99.0) if np.any(np.isfinite(depth_m)) else 50.0)
    ax2.imshow(d, cmap="turbo")
    ax2.set_title("Depth (m)")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB))
    ax3.set_title("Segmentation")
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 2, 4)
    if pc.shape[0] > 0:
        ax4.scatter(pc[:, 0], pc[:, 1], s=0.4, c=c.astype(np.float32) / 255.0, alpha=0.8, linewidths=0)
    ax4.set_title("Semantic Point Cloud (Top View)")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.grid(True, alpha=0.2)
    ax4.set_aspect("equal", adjustable="box")

    fig.suptitle(f"Sample {sample_id:02d} | Weather: {weather_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _available_weather_params() -> Dict[str, int]:
    out: Dict[str, int] = {}
    canonical = [
        "Rain", "Roadwetness", "Snow", "RoadSnow",
        "MapleLeaf", "RoadLeaf", "Dust", "Fog",
    ]
    for n in canonical:
        if hasattr(airsim.WeatherParameter, n):
            out[n] = int(getattr(airsim.WeatherParameter, n))

    # Fill extra params from current AirSim build (if any).
    for n in dir(airsim.WeatherParameter):
        if n.startswith("_") or n in out or n=="Enabled":
            continue
        v = getattr(airsim.WeatherParameter, n)
        if isinstance(v, (int, float)):
            out[n] = int(v)
    return out


def _weather_presets() -> List[Tuple[str, Dict[str, float]]]:
    return [
        ("clear", {}),
        # Particle-heavy extreme weather presets
        ("heavy_rain_particle", {"Rain": 0.85, "Roadwetness": 0.85}),
        ("storm_fog", {"Rain": 0.75, "Fog": 0.75, "Roadwetness": 0.9}),
        ("dense_fog", {"Fog": 0.5}),
        ("snowfall_particle", {"Snow": 0.85, "RoadSnow": 0.75}),
        ("blizzard_fog", {"Snow": 0.85, "Fog": 0.7, "RoadSnow": 0.8}),
        ("dust_particle", {"Dust": 0.95}),
        ("dust_fog", {"Dust": 0.8, "Fog": 0.55}),
        ("leaf_particle", {"MapleLeaf": 0.95, "RoadLeaf": 0.85}),
        ("rain_snow_mix", {"Rain": 0.55, "Snow": 0.55, "Roadwetness": 0.65, "RoadSnow": 0.4}),
    ]


def _select_scene_weather_group() -> List[Tuple[str, Dict[str, float]]]:
    """Fixed 4-weather bundle for same-scene comparison (particle-system focused)."""
    wanted = ["clear", "heavy_rain_particle", "snowfall_particle", "dust_particle"]
    presets = dict(_weather_presets())
    out: List[Tuple[str, Dict[str, float]]] = []
    for n in wanted:
        if n in presets:
            out.append((n, presets[n]))
    if len(out) < 4:
        # fallback: pad from preset list
        for n, cfg in _weather_presets():
            if n not in [x[0] for x in out]:
                out.append((n, cfg))
            if len(out) >= 4:
                break
    return out[:4]


def _apply_weather(
    client: airsim.MultirotorClient,
    available: Dict[str, int],
    weather_values: Dict[str, float],
):
    # Unified deterministic path:
    # enable weather -> reset all params -> apply target params.
    # "clear" is represented by empty weather_values (all params stay 0).
    client.simEnableWeather(True)
    for k, p in available.items():
        try:
            client.simSetWeatherParameter(p, 0.0)
        except Exception:
            pass
    for k, v in weather_values.items():
        if k not in available:
            continue
        try:
            client.simSetWeatherParameter(available[k], float(v))
        except Exception:
            pass


def _depth_to_vis(depth_m: np.ndarray) -> np.ndarray:
    d = depth_m.copy()
    d[~np.isfinite(d)] = 0.0
    if np.any(d > 0):
        p99 = np.percentile(d[d > 0], 99)
    else:
        p99 = 50.0
    d_clip = np.clip(d, 0.0, p99)
    d_norm = (d_clip / (d_clip.max() + 1e-6) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)


def _save_scene_comparison(scene_dir: str, weather_entries: List[Dict[str, object]]):
    n = len(weather_entries)
    if n == 0:
        return

    fig = plt.figure(figsize=(4.2 * n, 14))
    for i, e in enumerate(weather_entries):
        weather_name = str(e["weather_name"])
        rgb = e["rgb"]
        depth_m = e["depth_m"]
        seg = e["seg"]
        points_xyz = e["points_xyz"]
        sem_rgb = e["sem_rgb"]

        ax1 = fig.add_subplot(4, n, i + 1)
        ax1.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"{weather_name}\nRGB", fontsize=10)
        ax1.axis("off")

        ax2 = fig.add_subplot(4, n, n + i + 1)
        ax2.imshow(cv2.cvtColor(_depth_to_vis(depth_m), cv2.COLOR_BGR2RGB))
        ax2.set_title("Depth", fontsize=10)
        ax2.axis("off")

        ax3 = fig.add_subplot(4, n, 2 * n + i + 1)
        ax3.imshow(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB))
        ax3.set_title("Segmentation", fontsize=10)
        ax3.axis("off")

        ax4 = fig.add_subplot(4, n, 3 * n + i + 1)
        if points_xyz.shape[0] > 0:
            if points_xyz.shape[0] > 35000:
                idx = np.random.choice(points_xyz.shape[0], 35000, replace=False)
                pc = points_xyz[idx]
                c = sem_rgb[idx]
            else:
                pc = points_xyz
                c = sem_rgb
            ax4.scatter(pc[:, 0], pc[:, 1], s=0.4, c=c.astype(np.float32) / 255.0, alpha=0.8, linewidths=0)
        ax4.set_title("Semantic PC (Top)", fontsize=10)
        ax4.set_xlabel("X")
        ax4.set_ylabel("Y")
        ax4.grid(True, alpha=0.2)
        ax4.set_aspect("equal", adjustable="box")

    fig.suptitle("Same Scene, Different Weathers Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(scene_dir, "compare_modalities.png"), dpi=180)
    plt.close(fig)


def _save_summary_grid(image_paths: List[str], image_names: List[str], out_png: str):
    n = len(image_paths)
    cols = 5
    rows = int(math.ceil(n / cols))
    fig = plt.figure(figsize=(4 * cols, 3.5 * rows))
    for i, p in enumerate(image_paths):
        ax = fig.add_subplot(rows, cols, i + 1)
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(image_names[i], fontsize=9)
        ax.axis("off")
    fig.suptitle("Scene Comparison Overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Collect 8 scenes; each scene has 4 weather variants with comparisons.")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--simulator_tool_port", type=int, default=30000, help="AirVLNSimulatorServerTool msgpack port")
    parser.add_argument("--scene_id", type=int, default=1, help="Scene id to reopen via simulator tool")
    parser.add_argument("--no_reopen_scene", action="store_true", help="Do not call reopen_scenes on simulator tool")
    parser.add_argument("--sleep_after_reopen", type=float, default=2.5)
    parser.add_argument("--vehicle_name", type=str, default="Drone_LC_1")
    parser.add_argument("--lidar_vehicle_name", type=str, default="", help="If empty, use vehicle_name")
    parser.add_argument("--camera_id", type=str, default="front_0")
    parser.add_argument("--lidar_name", type=str, default="lidar_0")
    parser.add_argument("--use_lidar_depth", action="store_true", default=True)
    parser.add_argument("--no_lidar_depth", action="store_true")
    parser.add_argument("--num_scenes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="output/weather_semantic_samples")
    parser.add_argument("--param_py", type=str, default="src/common/param.py")
    parser.add_argument("--target_h", type=int, default=0, help="0 means read default from param.py")
    parser.add_argument("--target_w", type=int, default=0, help="0 means read default from param.py")
    parser.add_argument("--fov_deg", type=float, default=90.0)
    parser.add_argument("--min_depth", type=float, default=0.2)
    parser.add_argument("--max_depth", type=float, default=60.0)
    parser.add_argument("--sleep_after_weather", type=float, default=1.6)
    parser.add_argument("--scene_xy_jitter_m", type=float, default=10.0)
    parser.add_argument("--scene_yaw_step_deg", type=float, default=45.0)
    args = parser.parse_args()
    if args.no_lidar_depth:
        args.use_lidar_depth = False

    random.seed(args.seed)
    np.random.seed(args.seed)

    param_defaults = _read_image_shape_from_param_py(args.param_py)
    target_h = int(args.target_h) if int(args.target_h) > 0 else int(param_defaults["Image_Height_DEPTH"])
    target_w = int(args.target_w) if int(args.target_w) > 0 else int(param_defaults["Image_Width_DEPTH"])

    requested_port = int(args.port) if args.port is not None else None
    reopened_scene = False
    reopened_airsim_port = None
    if not args.no_reopen_scene:
        try:
            reopened_airsim_port = _reopen_scene_via_server_tool(
                host=args.ip,
                simulator_tool_port=int(args.simulator_tool_port),
                scene_id=int(args.scene_id),
            )
            if reopened_airsim_port is not None:
                requested_port = int(reopened_airsim_port)
                reopened_scene = True
                print(f"[SimulatorTool] reopened scene {args.scene_id}, AirSim port={requested_port}")
                time.sleep(max(0.0, float(args.sleep_after_reopen)))
            else:
                print(f"[SimulatorTool] reopen_scenes returned no AirSim port. Fallback to direct connect candidates.")
        except Exception as e:
            print(f"[SimulatorTool] reopen_scenes failed: {e}. Fallback to direct connect candidates.")

    out_root = args.out_dir
    os.makedirs(out_root, exist_ok=True)

    client, connected_port = _make_client(args.ip, requested_port)
    resolved_vehicle = _resolve_vehicle_name(client, args.vehicle_name, fallback="Drone_1")
    resolved_lidar_vehicle = args.lidar_vehicle_name.strip() if args.lidar_vehicle_name.strip() else resolved_vehicle
    resolved_lidar_vehicle = _resolve_vehicle_name(client, resolved_lidar_vehicle, fallback=resolved_vehicle)
    try:
        client.enableApiControl(True, vehicle_name=resolved_vehicle)
        client.armDisarm(True, vehicle_name=resolved_vehicle)
    except Exception:
        pass

    base_pose = client.simGetVehiclePose(vehicle_name=resolved_vehicle)
    base_x = float(base_pose.position.x_val)
    base_y = float(base_pose.position.y_val)
    base_z = float(base_pose.position.z_val if abs(base_pose.position.z_val) > 1e-3 else -8.0)

    w_params = _available_weather_params()

    dataset_meta = {
        "created_at": datetime.now().isoformat(),
        "param_py": args.param_py,
        "param_defaults": param_defaults,
        "ip": args.ip,
        "scene_id": int(args.scene_id),
        "simulator_tool_port": int(args.simulator_tool_port),
        "reopened_scene": bool(reopened_scene),
        "reopened_airsim_port": reopened_airsim_port,
        "requested_port": requested_port,
        "connected_port": connected_port,
        "vehicle_name": resolved_vehicle,
        "lidar_vehicle_name": resolved_lidar_vehicle,
        "camera_id": args.camera_id,
        "lidar_name": args.lidar_name,
        "depth_mode": "lidar_raytrace_projection" if args.use_lidar_depth else "depth_perspective",
        "particle_weather_intent": True,
        "num_scenes": int(args.num_scenes),
        "weathers_per_scene": [w[0] for w in _select_scene_weather_group()],
        "target_hw": [target_h, target_w],
        "fov_deg_requested": float(args.fov_deg),
        "min_depth": float(args.min_depth),
        "max_depth": float(args.max_depth),
        "scenes": [],
    }

    scene_compare_paths: List[str] = []
    scene_compare_names: List[str] = []
    scene_weather_group = _select_scene_weather_group()

    for scene_idx in range(args.num_scenes):
        scene_dir = os.path.join(out_root, f"scene_{scene_idx:02d}")
        os.makedirs(scene_dir, exist_ok=True)

        dx = float(np.random.uniform(-args.scene_xy_jitter_m, args.scene_xy_jitter_m))
        dy = float(np.random.uniform(-args.scene_xy_jitter_m, args.scene_xy_jitter_m))
        yaw_deg = float((scene_idx * args.scene_yaw_step_deg) % 360.0)
        scene_pose = _set_vehicle_pose_xyz_yaw(
            client,
            x=base_x + dx,
            y=base_y + dy,
            z=base_z,
            yaw_deg=yaw_deg,
            vehicle_name=resolved_vehicle,
        )

        scene_meta = {
            "scene_id": int(scene_idx),
            "scene_pose_xyzq": _pose_to_list(scene_pose),
            "weathers": [],
        }
        weather_entries: List[Dict[str, object]] = []

        for weather_idx, (weather_name, weather_cfg) in enumerate(scene_weather_group):
            weather_dir = os.path.join(scene_dir, f"weather_{weather_idx:02d}_{weather_name}")
            os.makedirs(weather_dir, exist_ok=True)

            # Re-apply same pose for each weather to guarantee same frame/range.
            _set_vehicle_pose_xyz_yaw(
                client,
                x=base_x + dx,
                y=base_y + dy,
                z=base_z,
                yaw_deg=yaw_deg,
                vehicle_name=resolved_vehicle,
            )

            _apply_weather(client, w_params, weather_cfg)
            print(f"[scene {scene_idx:02d}] apply weather '{weather_name}': {weather_cfg}")
            time.sleep(max(0.0, float(args.sleep_after_weather)))

            cam_fov_deg = _get_camera_fov_deg(client, camera_id=args.camera_id, vehicle_name=resolved_vehicle, fallback=float(args.fov_deg))
            if args.use_lidar_depth:
                rgb_raw, depth_raw, seg_raw, lidar_meta = _capture_rgb_seg_with_lidar_depth(
                    client=client,
                    vehicle_name=resolved_vehicle,
                    camera_id=args.camera_id,
                    lidar_vehicle_name=resolved_lidar_vehicle,
                    lidar_name=args.lidar_name,
                    fov_deg=float(cam_fov_deg),
                    min_depth=float(args.min_depth),
                    max_depth=float(args.max_depth),
                )
            else:
                rgb_raw, depth_raw, seg_raw = _capture_rgb_depth_seg(client, vehicle_name=resolved_vehicle, camera_id=args.camera_id)
                lidar_meta = {
                    "depth_source": "depth_perspective",
                    "lidar_vehicle": resolved_lidar_vehicle,
                    "lidar_name": args.lidar_name,
                    "lidar_points_raw": 0,
                    "lidar_points_used": 0,
                }
            rgb, depth_m, seg, align_meta = _align_modalities(
                rgb=rgb_raw,
                depth_m=depth_raw,
                seg=seg_raw,
                target_h=target_h,
                target_w=target_w,
            )
            cam_pose = _get_camera_pose(client, camera_id=args.camera_id, vehicle_name=resolved_vehicle)

            sem_pc = _semantic_point_cloud(
                rgb=rgb,
                depth_m=depth_m,
                seg=seg,
                camera_pose=cam_pose,
                fov_deg=float(cam_fov_deg),
                min_depth=float(args.min_depth),
                max_depth=float(args.max_depth),
            )

            rgb_path = os.path.join(weather_dir, "rgb.png")
            seg_path = os.path.join(weather_dir, "segmentation.png")
            depth_npy_path = os.path.join(weather_dir, "depth_m.npy")
            depth_vis_path = os.path.join(weather_dir, "depth_vis.png")
            pc_npz_path = os.path.join(weather_dir, "semantic_pointcloud.npz")
            pc_ply_path = os.path.join(weather_dir, "semantic_pointcloud.ply")
            viz_path = os.path.join(weather_dir, "overview.png")

            cv2.imwrite(rgb_path, rgb)
            cv2.imwrite(seg_path, seg)
            np.save(depth_npy_path, depth_m.astype(np.float32))
            cv2.imwrite(depth_vis_path, _depth_to_vis(depth_m))

            np.savez_compressed(
                pc_npz_path,
                points_xyz=sem_pc["points_xyz"],
                points_rgb=sem_pc["points_rgb"],
                points_sem_rgb=sem_pc["points_sem_rgb"],
                points_sem_id=sem_pc["points_sem_id"],
            )
            _write_ascii_ply_semantic(pc_ply_path, sem_pc["points_xyz"], sem_pc["points_sem_rgb"])

            _save_visualization(
                out_png=viz_path,
                rgb=rgb,
                depth_m=depth_m,
                seg=seg,
                points_xyz=sem_pc["points_xyz"],
                sem_rgb=sem_pc["points_sem_rgb"],
                weather_name=weather_name,
                sample_id=weather_idx,
            )

            uniq_ids = np.unique(sem_pc["points_sem_id"])
            weather_meta = {
                "weather_name": weather_name,
                "weather_values": weather_cfg,
                "camera_pose_xyzq": _pose_to_list(cam_pose),
                "camera_fov_deg": float(cam_fov_deg),
                "alignment": align_meta,
                "depth_source": lidar_meta.get("depth_source", "unknown"),
                "lidar_meta": lidar_meta,
                "image_shape_hw3": [int(rgb.shape[0]), int(rgb.shape[1]), int(rgb.shape[2])],
                "num_points": int(sem_pc["points_xyz"].shape[0]),
                "num_unique_semantic_ids": int(len(uniq_ids)),
                "outputs": {
                    "rgb": os.path.relpath(rgb_path, out_root),
                    "segmentation": os.path.relpath(seg_path, out_root),
                    "depth_npy": os.path.relpath(depth_npy_path, out_root),
                    "depth_vis": os.path.relpath(depth_vis_path, out_root),
                    "semantic_pointcloud_npz": os.path.relpath(pc_npz_path, out_root),
                    "semantic_pointcloud_ply": os.path.relpath(pc_ply_path, out_root),
                    "overview": os.path.relpath(viz_path, out_root),
                },
            }
            with open(os.path.join(weather_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(weather_meta, f, ensure_ascii=False, indent=2)

            scene_meta["weathers"].append(weather_meta)
            weather_entries.append(
                {
                    "weather_name": weather_name,
                    "rgb": rgb,
                    "depth_m": depth_m,
                    "seg": seg,
                    "points_xyz": sem_pc["points_xyz"],
                    "sem_rgb": sem_pc["points_sem_rgb"],
                }
            )

        _save_scene_comparison(scene_dir, weather_entries)
        scene_meta["compare_modalities"] = os.path.relpath(os.path.join(scene_dir, "compare_modalities.png"), out_root)
        with open(os.path.join(scene_dir, "scene_meta.json"), "w", encoding="utf-8") as f:
            json.dump(scene_meta, f, ensure_ascii=False, indent=2)

        dataset_meta["scenes"].append(scene_meta)
        scene_compare_paths.append(os.path.join(scene_dir, "compare_modalities.png"))
        scene_compare_names.append(f"scene_{scene_idx:02d}")
        print(f"[{scene_idx + 1}/{args.num_scenes}] collected scene_{scene_idx:02d} with 4 weathers -> {scene_dir}")

    _save_summary_grid(scene_compare_paths, scene_compare_names, os.path.join(out_root, "summary_grid.png"))
    with open(os.path.join(out_root, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, ensure_ascii=False, indent=2)

    try:
        client.simEnableWeather(False)
    except Exception:
        pass

    print(f"\nDone. Dataset saved to: {os.path.abspath(out_root)}")
    print("Key files:")
    print(f"- {os.path.join(out_root, 'dataset_meta.json')}")
    print(f"- {os.path.join(out_root, 'summary_grid.png')}")


if __name__ == "__main__":
    main()
