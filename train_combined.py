#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import io
import inspect
import json
import math
import os
import pickle
import random
import struct
import time
import zipfile
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR, StepLR

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from torch.amp import autocast as _autocast
    from torch.amp import GradScaler as _GradScaler

    _AMP_REQUIRES_DEVICE_TYPE = True
except ImportError:
    from torch.cuda.amp import autocast as _autocast
    from torch.cuda.amp import GradScaler as _GradScaler

    _AMP_REQUIRES_DEVICE_TYPE = False

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

from MapGlow11_27_original import Glow as OriginalGlow
from RouteMixtureMapGlow import Glow as RouteMixtureGlow
# Backward-compatible injection point used by lightweight integration tests and
# external launchers that replace the original model class.
Glow = OriginalGlow
from trajectory_representation import (
    decode_constant_velocity_residual_torch,
    decode_state_deltas_torch,
    encode_constant_velocity_residual_np,
    encode_state_deltas_np,
)


class _NumpyCoreCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_npz_object_array_compat(npz_path, member_name):
    member_file = f"{member_name}.npy"
    with zipfile.ZipFile(npz_path, "r") as archive:
        with archive.open(member_file, "r") as handle:
            if handle.read(6) != b"\x93NUMPY":
                raise ValueError(f"{member_file} is not a valid .npy payload")

            major, minor = handle.read(2)
            if major == 1:
                header_len = struct.unpack("<H", handle.read(2))[0]
            elif major in (2, 3):
                header_len = struct.unpack("<I", handle.read(4))[0]
            else:
                raise ValueError(f"unsupported npy version {major}.{minor} for {member_file}")

            handle.read(header_len)
            payload = io.BytesIO(handle.read())
            return _NumpyCoreCompatUnpickler(payload).load()


def load_scene_stats(scene_stats_raw, num_scenes):
    scene_stats = []
    for item in scene_stats_raw:
        center = np.asarray(item["mean"], dtype=np.float32)
        scale = np.float32(item["scale"])
        scene_stats.append(np.concatenate([center, np.array([scale], dtype=np.float32)]))
    return np.stack(scene_stats, axis=0).astype(np.float32)


def _first_present_array(data, names, dtype=None):
    """Return the first named NPZ array, without silently accepting bad shapes."""
    for name in names:
        if name in data.files:
            value = np.asarray(data[name])
            return value.astype(dtype, copy=False) if dtype is not None else value
    return None


def _checked_bool_mask(value, expected_shape, name):
    if value is None:
        return None
    value = np.asarray(value, dtype=bool)
    if tuple(value.shape) != tuple(expected_shape):
        raise ValueError(
            f"{name} must have shape {tuple(expected_shape)}, got {tuple(value.shape)}"
        )
    return value


def load_normalization_stats(data, num_scenes, combined_path):
    """Load numeric normalization metadata first, with old object scene_stats fallback."""
    center = _first_present_array(data, ("normalization_center", "scene_center"), np.float32)
    scale = _first_present_array(data, ("normalization_scale", "position_scale"), np.float32)
    if center is not None or scale is not None:
        if center is None or scale is None:
            raise ValueError(
                "normalization_center and normalization_scale must either both be present or both be absent"
            )
        center = np.asarray(center, dtype=np.float32)
        scale = np.asarray(scale, dtype=np.float32).reshape(-1)
        if center.shape != (num_scenes, 2) or scale.shape != (num_scenes,):
            raise ValueError(
                "normalization metadata must have shapes "
                f"({num_scenes}, 2) and ({num_scenes},), got {center.shape} and {scale.shape}"
            )
        if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("normalization metadata must be finite and all scales must be positive")
        return np.concatenate([center, scale[:, None]], axis=1).astype(np.float32)

    try:
        scene_stats_raw = data["scene_stats"]
    except ModuleNotFoundError as exc:
        print(f"[dataset] scene_stats pickle compatibility fallback: {exc}")
        scene_stats_raw = load_npz_object_array_compat(combined_path, "scene_stats")
    except KeyError:
        print("[dataset] warning: no normalization metadata; scene_stats defaults to zeros")
        return np.zeros((num_scenes, 3), dtype=np.float32)

    try:
        return load_scene_stats(scene_stats_raw, num_scenes)
    except Exception as exc:
        print(f"[dataset] warning: failed to load scene_stats, fallback to zeros. reason: {exc}")
        return np.zeros((num_scenes, 3), dtype=np.float32)


def gather_last_valid_dimensions(dimensions, timestep_mask):
    """Gather normalized length/width at each agent's last observed timestep."""
    dimensions = np.asarray(dimensions, dtype=np.float32)
    timestep_mask = np.asarray(timestep_mask, dtype=bool)
    if dimensions.ndim != 4 or dimensions.shape[1] != 2:
        raise ValueError(f"dimensions must be [N,2,T,V], got {dimensions.shape}")
    expected = (dimensions.shape[0], dimensions.shape[2], dimensions.shape[3])
    if timestep_mask.shape != expected:
        raise ValueError(f"dimension timestep_mask must be {expected}, got {timestep_mask.shape}")

    valid_any = timestep_mask.any(axis=1)
    reverse_idx = timestep_mask[:, ::-1, :].argmax(axis=1)
    last_idx = dimensions.shape[2] - 1 - reverse_idx
    gather_idx = last_idx[:, None, None, :]
    gathered = np.take_along_axis(dimensions, gather_idx, axis=2).squeeze(2)
    return np.where(valid_any[:, None, :], gathered, 0.0).astype(np.float32)


def wrap_to_pi_np(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def infer_target_labels(
    target_data, target_vehicle_mask, turn_angle_threshold_deg=30.0,
    stationary_dist_threshold=0.0, yaw_scale=np.pi,
):
    labels = np.full(target_vehicle_mask.shape, 4, dtype=np.int64)
    turn_threshold = np.deg2rad(float(turn_angle_threshold_deg))

    # target_data uses -1.0 on every channel for padded timesteps.
    timestep_valid = ~np.isclose(target_data[:, :5], -1.0, atol=0.05).all(axis=1)
    num_scenes, _, _, num_agents = target_data.shape
    for scene_idx in range(num_scenes):
        for agent_idx in range(num_agents):
            if not target_vehicle_mask[scene_idx, agent_idx]:
                continue
            valid_t = timestep_valid[scene_idx, :, agent_idx]
            if valid_t.sum() < 2:
                continue

            xy = target_data[scene_idx, :2, valid_t, agent_idx].T
            yaw = target_data[scene_idx, 4, valid_t, agent_idx] * float(yaw_scale)
            displacement = float(np.linalg.norm(xy[-1] - xy[0]))
            if stationary_dist_threshold > 0.0 and displacement < stationary_dist_threshold:
                labels[scene_idx, agent_idx] = 3
                continue

            yaw_unwrapped = np.unwrap(yaw.astype(np.float64))
            heading_delta = float(wrap_to_pi_np(yaw_unwrapped[-1] - yaw_unwrapped[0]))
            if heading_delta > turn_threshold:
                labels[scene_idx, agent_idx] = 1
            elif heading_delta < -turn_threshold:
                labels[scene_idx, agent_idx] = 2
            else:
                labels[scene_idx, agent_idx] = 0
    return labels


def uses_label_condition(train_mode, label_source="auto"):
    return train_mode == "initialization" and label_source != "none"


def make_grad_scaler(enabled, init_scale=256.0):
    init_scale = float(init_scale)
    if _AMP_REQUIRES_DEVICE_TYPE:
        return _GradScaler("cuda", enabled=enabled, init_scale=init_scale)
    return _GradScaler(enabled=enabled, init_scale=init_scale)


def cuda_autocast(enabled):
    if _AMP_REQUIRES_DEVICE_TYPE:
        return _autocast("cuda", enabled=enabled)
    return _autocast(enabled=enabled)


def get_env_int(name, default=None):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def set_random_seed(seed, rank=0):
    if seed is None or seed < 0:
        return
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_model(model):
    model = getattr(model, "module", model)
    return getattr(model, "_orig_mod", model)


def normalize_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    prefixes = ("module.", "_orig_mod.")
    normalized = state_dict
    for prefix in prefixes:
        if normalized and all(isinstance(k, str) and k.startswith(prefix) for k in normalized.keys()):
            normalized = {k[len(prefix):]: v for k, v in normalized.items()}
    return normalized


def torch_load_checkpoint(path, map_location="cpu", weights_only=False):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def capture_rng_state():
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(checkpoint):
    """Restore every process-level RNG recorded by ``capture_rng_state``."""
    if not checkpoint:
        return False
    restored = False
    if checkpoint.get("python_rng_state") is not None:
        random.setstate(checkpoint["python_rng_state"])
        restored = True
    if checkpoint.get("numpy_rng_state") is not None:
        np.random.set_state(checkpoint["numpy_rng_state"])
        restored = True
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        restored = True
    cuda_state = checkpoint.get("cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
        restored = True
    return restored


def load_model_state_checked(model, state_dict, path, allow_partial=False):
    state_dict = normalize_state_dict_keys(state_dict)
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint model state at {path} is not a state dict")
    if allow_partial:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(
                f"[checkpoint] partial model load explicitly enabled: {path}; "
                f"missing={missing}, unexpected={unexpected}"
            )
        return missing, unexpected
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint is incompatible with the current model: {path}. "
            "Use --allow_partial_checkpoint only for an intentional architecture migration."
        ) from exc
    return [], []


def save_training_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    args,
    step,
    epoch,
    epoch_step,
    steps_per_epoch,
    extra_state=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    next_epoch_step = int(epoch_step) + 1
    next_epoch = int(epoch)
    if next_epoch_step >= int(steps_per_epoch):
        next_epoch += 1
        next_epoch_step = 0
    payload = {
        "format_version": 4,
        "step": int(step),
        "global_step": int(step),
        "next_iter": int(step) + 1,
        "epoch": int(epoch),
        "epoch_step": int(epoch_step),
        "next_epoch": int(next_epoch),
        "next_epoch_step": int(next_epoch_step),
        "steps_per_epoch": int(steps_per_epoch),
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "args": vars(args).copy(),
    }
    payload.update(capture_rng_state())
    if extra_state:
        payload.update(dict(extra_state))
    torch.save(payload, path)


def load_training_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    load_optimizer=True,
    allow_partial=False,
    restore_rng=False,
):
    if not path:
        return None
    if not os.path.exists(path):
        print(f"[checkpoint] missing file: {path}")
        return None

    ckpt = torch_load_checkpoint(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        saved_arch = (ckpt.get("args") or {}).get("model_arch", "mapglow")
        current_arch = getattr(unwrap_model(model), "model_arch", "mapglow")
        if saved_arch != current_arch:
            raise ValueError(
                f"checkpoint architecture mismatch: saved={saved_arch}, current={current_arch}"
            )
        missing, unexpected = load_model_state_checked(
            model, ckpt["model_state"], path, allow_partial=allow_partial
        )
        print(f"[checkpoint] loaded model: {path}; missing={len(missing)} unexpected={len(unexpected)}")
        if load_optimizer and optimizer is not None and ckpt.get("optimizer_state") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if load_optimizer and scheduler is not None and ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if load_optimizer and scaler is not None and ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        if restore_rng:
            restored = restore_rng_state(ckpt)
            print(f"[checkpoint] RNG state restored={restored}")
        return ckpt

    missing, unexpected = load_model_state_checked(model, ckpt, path, allow_partial=allow_partial)
    print(f"[checkpoint] loaded legacy state dict: {path}; missing={len(missing)} unexpected={len(unexpected)}")
    return None


def distributed_bad_flag(local_bad, is_distributed, device):
    flag = torch.tensor(int(bool(local_bad)), device=device, dtype=torch.int32)
    if is_distributed:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def broadcast_tensor_from_rank0_(tensor):
    if tensor.is_contiguous():
        dist.broadcast(tensor, src=0)
        return

    synced = tensor.contiguous()
    dist.broadcast(synced, src=0)
    tensor.copy_(synced)


def distributed_label_condition(train_mode, label_source, keep_prob, is_distributed, device):
    if not uses_label_condition(train_mode, label_source):
        return False
    keep = torch.rand(1, device=device) < float(keep_prob)
    if is_distributed:
        dist.broadcast(keep, src=0)
    return bool(keep.item())


def valid_dimension_count(batch, channels):
    mask = batch.get("loss_timestep_mask", batch["timestep_mask"])
    valid_tv = mask.float().sum(dim=(1, 2))
    return (valid_tv * int(channels)).clamp_min(1.0)


def gradients_are_finite(parameters):
    return all(
        torch.isfinite(parameter.grad).all().item()
        for parameter in parameters
        if parameter.grad is not None
    )


def build_lr_scheduler(optimizer, args, steps_per_epoch):
    scheduler_type = getattr(args, "scheduler_type", "step")
    if scheduler_type == "step":
        return StepLR(optimizer, step_size=10000, gamma=0.96)
    if scheduler_type != "cosine":
        raise ValueError(f"unsupported scheduler_type: {scheduler_type}")

    configured_max = int(getattr(args, "max_steps", -1))
    total_steps = (
        configured_max
        if configured_max > 0
        else max(1, int(args.epochs) * int(steps_per_epoch))
    )
    warmup_steps = min(
        max(0, int(getattr(args, "warmup_steps", 0))),
        max(0, total_steps - 1),
    )
    min_ratio = float(getattr(args, "min_lr_ratio", 0.05))

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


class CombinedInteractionDataset(Dataset):
    def __init__(
        self,
        combined_path,
        in_channel=7,
        train_mode="initialization",
        history_steps=10,
        future_steps=30,
        prediction_target_steps=32,
        prediction_representation="absolute",
        label_source="auto",
        turn_angle_threshold_deg=30.0,
        stationary_dist_threshold=0.0,
        allow_legacy_prediction_data=False,
        require_map_topology=False,
    ):
        if in_channel not in (5, 7):
            raise ValueError(f"in_channel must be 5 or 7, got {in_channel}")
        if train_mode not in ("initialization", "prediction"):
            raise ValueError(f"train_mode must be initialization or prediction, got {train_mode}")
        if label_source not in ("auto", "none", "dataset", "target"):
            raise ValueError(f"label_source must be auto, none, dataset, or target, got {label_source}")
        if prediction_representation not in ("absolute", "delta", "cv_residual"):
            raise ValueError(
                "prediction_representation must be absolute, delta, or cv_residual, got "
                f"{prediction_representation}"
            )
        if (
            train_mode == "prediction"
            and prediction_representation in ("delta", "cv_residual")
            and in_channel != 5
        ):
            raise ValueError(
                "prediction residuals represent the five dynamic trajectory channels; "
                "use in_channel=5"
            )

        data = np.load(combined_path, allow_pickle=True)
        has_safe_contract = all(
            key in data.files
            for key in (
                "context_agent_mask",
                "history_timestep_mask",
                "future_timestep_mask",
            )
        )
        forecasting_safe = (
            bool(np.asarray(data["forecasting_safe"]).item())
            if "forecasting_safe" in data.files
            else False
        )
        if train_mode == "prediction" and (not has_safe_contract or not forecasting_safe):
            message = (
                "prediction NPZ lacks forecasting_safe=True and the complete explicit "
                "history/future mask contract; it may contain future-derived normalization "
                "or agent selection"
            )
            if not allow_legacy_prediction_data:
                data.close()
                raise ValueError(
                    message
                    + ". Regenerate it with data_preprocess.py --forecasting_safe, or pass "
                    "--allow_legacy_prediction_data only for a knowingly leaky legacy run."
                )
            print(f"[dataset] HIGH-RISK LEGACY OVERRIDE: {message}")

        trajectories = data["trajectories"].astype(np.float32)  # [N, 5, T, V]
        dimensions = data["dimensions"].astype(np.float32)      # [N, 2, T, V]
        labels = data["labels"].astype(np.int64)
        agent_types = data["agent_types"].astype(np.int64)
        map_data = data["map_data"].astype(np.float32)          # [N, L, P, 2]
        map_mask = data["map_mask"].astype(bool)
        map_type = data["map_type"].astype(np.int64)
        map_speed_limit = data["map_speed_limit"].astype(np.float32)
        topology_keys = (
            "map_lanelet_id", "map_lane_subtype", "map_left_boundary_type",
            "map_right_boundary_type", "lane_edge_index", "lane_edge_type", "lane_edge_mask",
        )
        missing_topology = [key for key in topology_keys if key not in data.files]
        if require_map_topology and missing_topology:
            data.close()
            raise ValueError(
                f"route_mixture_glow requires forecasting-safe v5 topology fields: {missing_topology}"
            )
        max_edges = int(np.asarray(data["max_lane_edges"]).item()) if "max_lane_edges" in data.files else 1
        map_lanelet_id = data["map_lanelet_id"].astype(np.int64) if "map_lanelet_id" in data.files else np.zeros_like(map_type)
        map_lane_subtype = data["map_lane_subtype"].astype(np.int64) if "map_lane_subtype" in data.files else np.zeros_like(map_type)
        map_left_boundary_type = data["map_left_boundary_type"].astype(np.int64) if "map_left_boundary_type" in data.files else np.zeros_like(map_type)
        map_right_boundary_type = data["map_right_boundary_type"].astype(np.int64) if "map_right_boundary_type" in data.files else np.zeros_like(map_type)
        lane_edge_index = data["lane_edge_index"].astype(np.int64) if "lane_edge_index" in data.files else np.zeros((len(map_type), max_edges, 2), dtype=np.int64)
        lane_edge_type = data["lane_edge_type"].astype(np.int64) if "lane_edge_type" in data.files else np.zeros((len(map_type), max_edges), dtype=np.int64)
        lane_edge_mask = data["lane_edge_mask"].astype(bool) if "lane_edge_mask" in data.files else np.zeros((len(map_type), max_edges), dtype=bool)
        map_names = data["map_names"]
        num_scenes, _, sequence_steps, num_agents = trajectories.shape
        scene_stats = load_normalization_stats(data, num_scenes, combined_path)
        velocity_scale_raw = _first_present_array(
            data, ("normalization_velocity_scale",), np.float32
        )
        yaw_scale_raw = _first_present_array(
            data, ("normalization_yaw_scale",), np.float32
        )
        velocity_scale = float(
            np.asarray(velocity_scale_raw).item()
            if velocity_scale_raw is not None else 15.0
        )
        yaw_scale = float(
            np.asarray(yaw_scale_raw).item()
            if yaw_scale_raw is not None else np.pi
        )
        if (
            not np.isfinite(velocity_scale) or velocity_scale <= 0
            or not np.isfinite(yaw_scale) or yaw_scale <= 0
        ):
            raise ValueError("velocity and yaw normalization scales must be finite and positive")
        position_scale = scene_stats[:, 2].copy()
        position_scale = np.where(position_scale > 0, position_scale, 50.0)
        state_scales = np.stack(
            [
                position_scale,
                np.full(num_scenes, velocity_scale, dtype=np.float32),
                np.full(num_scenes, yaw_scale, dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)

        if in_channel == 7:
            full_data = np.concatenate([trajectories, dimensions], axis=1)
        else:
            full_data = trajectories

        inferred_padding = (
            np.isclose(trajectories, 0.0, atol=0.0).all(axis=1)
            & np.isclose(dimensions, 0.0, atol=0.0).all(axis=1)
        )  # [N, T, V]
        full_timestep_mask = _checked_bool_mask(
            _first_present_array(data, ("timestep_mask", "trajectory_timestep_mask", "observation_mask")),
            (num_scenes, sequence_steps, num_agents),
            "timestep_mask",
        )
        if full_timestep_mask is None:
            full_timestep_mask = ~inferred_padding
        full_data = np.where(full_timestep_mask[:, None, :, :], full_data, -1.0)

        explicit_context_mask = _checked_bool_mask(
            _first_present_array(data, ("context_agent_mask", "vehicle_mask")),
            (num_scenes, num_agents),
            "context_agent_mask/vehicle_mask",
        )
        context_agent_mask = (
            explicit_context_mask.copy()
            if explicit_context_mask is not None
            else full_timestep_mask.any(axis=1)
        )
        target_vehicle_mask = full_timestep_mask.any(axis=1)
        history_vehicle_mask = np.zeros_like(target_vehicle_mask)
        history_timestep_mask = np.zeros((num_scenes, 0, num_agents), dtype=bool)

        if train_mode == "prediction":
            if history_steps <= 0:
                raise ValueError("prediction mode requires history_steps > 0")
            if future_steps <= 0:
                raise ValueError("prediction mode requires future_steps > 0")
            if prediction_target_steps < future_steps:
                raise ValueError(
                    "prediction_target_steps must be >= future_steps "
                    f"({prediction_target_steps} < {future_steps})"
                )
            if history_steps + future_steps > full_data.shape[2]:
                raise ValueError(
                    "history_steps + future_steps exceeds dataset sequence length: "
                    f"{history_steps} + {future_steps} > {full_data.shape[2]}"
                )

            history_data = full_data[:, :, :history_steps, :]
            future_data = full_data[:, :, history_steps:history_steps + future_steps, :]
            history_timestep_mask = _checked_bool_mask(
                _first_present_array(data, ("history_timestep_mask",)),
                (num_scenes, history_steps, num_agents),
                "history_timestep_mask",
            )
            if history_timestep_mask is None:
                history_timestep_mask = full_timestep_mask[:, :history_steps, :].copy()
            future_timestep_mask = _checked_bool_mask(
                _first_present_array(data, ("future_timestep_mask", "loss_timestep_mask")),
                (num_scenes, future_steps, num_agents),
                "future_timestep_mask",
            )
            if future_timestep_mask is None:
                future_timestep_mask = full_timestep_mask[
                    :, history_steps:history_steps + future_steps, :
                ].copy()
            if prediction_target_steps > future_steps:
                pad_shape = (
                    future_data.shape[0],
                    future_data.shape[1],
                    prediction_target_steps - future_steps,
                    future_data.shape[3],
                )
                pad_data = np.full(pad_shape, -1.0, dtype=future_data.dtype)
                target_data = np.concatenate([future_data, pad_data], axis=2)
                loss_timestep_mask = np.concatenate(
                    [
                        future_timestep_mask,
                        np.zeros(
                            (num_scenes, prediction_target_steps - future_steps, num_agents),
                            dtype=bool,
                        ),
                    ],
                    axis=1,
                )
            else:
                target_data = future_data
                loss_timestep_mask = future_timestep_mask
            history_data = np.where(history_timestep_mask[:, None, :, :], history_data, -1.0)
            target_data = np.where(loss_timestep_mask[:, None, :, :], target_data, -1.0)
            history_vehicle_mask = history_timestep_mask.any(axis=1)
            target_vehicle_mask = loss_timestep_mask.any(axis=1)
            if explicit_context_mask is None:
                context_agent_mask = history_vehicle_mask.copy()
            unsupported_future = loss_timestep_mask & ~context_agent_mask[:, None, :]
            dropped_future_points = int(unsupported_future.sum())
            dropped_future_agents = int(unsupported_future.any(axis=1).sum())
            if dropped_future_points:
                print(
                    "[dataset] WARNING: dropping supervised future observations outside the "
                    f"history-visible support: agents={dropped_future_agents}, "
                    f"timesteps={dropped_future_points}"
                )
                loss_timestep_mask = loss_timestep_mask & context_agent_mask[:, None, :]
                target_data = np.where(
                    loss_timestep_mask[:, None, :, :], target_data, -1.0
                )
                target_vehicle_mask = loss_timestep_mask.any(axis=1)
            if prediction_representation == "delta":
                target_data = encode_state_deltas_np(
                    target_data,
                    history_data,
                    loss_timestep_mask,
                    history_timestep_mask,
                    yaw_scale=yaw_scale,
                )
            elif prediction_representation == "cv_residual":
                target_data = encode_constant_velocity_residual_np(
                    target_data,
                    history_data,
                    loss_timestep_mask,
                    history_timestep_mask,
                    yaw_scale=yaw_scale,
                )
        else:
            history_data = full_data[:, :, :0, :]
            target_data = full_data
            loss_timestep_mask = full_timestep_mask.copy()

        # Prediction support is history-defined: future-only observations are
        # removed above so training and sampling cover exactly the same agents.
        context_agent_mask = np.asarray(context_agent_mask, dtype=bool)
        static_source_mask = history_timestep_mask if train_mode == "prediction" else full_timestep_mask
        static_source_dimensions = (
            dimensions[:, :, :history_steps, :]
            if train_mode == "prediction"
            else dimensions
        )
        static_dimensions = _first_present_array(data, ("static_dimensions",), np.float32)
        if static_dimensions is not None:
            if static_dimensions.shape != (num_scenes, 2, num_agents):
                raise ValueError(
                    f"static_dimensions must be {(num_scenes, 2, num_agents)}, "
                    f"got {static_dimensions.shape}"
                )
        else:
            static_dimensions = gather_last_valid_dimensions(
                static_source_dimensions, static_source_mask
            )

        label_source_used = label_source
        if train_mode == "prediction":
            label_source_used = "none"
        elif label_source_used == "auto":
            label_source_used = "dataset"
        if label_source_used == "target":
            labels = infer_target_labels(
                target_data,
                target_vehicle_mask,
                turn_angle_threshold_deg=turn_angle_threshold_deg,
                stationary_dist_threshold=stationary_dist_threshold,
                yaw_scale=yaw_scale,
            )

        self.history_data = torch.from_numpy(history_data)
        self.target_data = torch.from_numpy(target_data)
        self.labels = torch.from_numpy(labels)
        self.agent_types = torch.from_numpy(agent_types)
        self.map_data = torch.from_numpy(map_data)
        self.map_mask = torch.from_numpy(map_mask)
        self.map_type = torch.from_numpy(map_type)
        self.map_speed_limit = torch.from_numpy(map_speed_limit)
        self.map_lanelet_id = torch.from_numpy(map_lanelet_id)
        self.map_lane_subtype = torch.from_numpy(map_lane_subtype)
        self.map_left_boundary_type = torch.from_numpy(map_left_boundary_type)
        self.map_right_boundary_type = torch.from_numpy(map_right_boundary_type)
        self.lane_edge_index = torch.from_numpy(lane_edge_index)
        self.lane_edge_type = torch.from_numpy(lane_edge_type)
        self.lane_edge_mask = torch.from_numpy(lane_edge_mask)
        self.scene_stats = torch.from_numpy(scene_stats)
        self.state_scales = torch.from_numpy(state_scales)
        self.target_vehicle_mask = torch.from_numpy(target_vehicle_mask)
        self.context_agent_mask = torch.from_numpy(context_agent_mask)
        self.history_vehicle_mask = torch.from_numpy(history_vehicle_mask)
        self.loss_timestep_mask = torch.from_numpy(loss_timestep_mask)
        self.history_timestep_mask = torch.from_numpy(history_timestep_mask)
        self.static_dimensions = torch.from_numpy(static_dimensions)
        self.map_names = map_names

        self.input_channels = int(target_data.shape[1])
        self.time_steps = int(target_data.shape[2])
        self.max_agents = int(target_data.shape[3])
        self.max_lanes = int(map_data.shape[1])
        self.max_points = int(map_data.shape[2])
        self.train_mode = train_mode
        self.history_steps = int(history_steps if train_mode == "prediction" else 0)
        self.future_steps = int(future_steps if train_mode == "prediction" else 0)
        self.prediction_target_steps = int(prediction_target_steps if train_mode == "prediction" else 0)
        self.prediction_representation = (
            prediction_representation if train_mode == "prediction" else "absolute"
        )
        self.label_source = label_source_used
        self.dropped_future_only_points = (
            dropped_future_points if train_mode == "prediction" else 0
        )
        self.dropped_future_only_agents = (
            dropped_future_agents if train_mode == "prediction" else 0
        )
        data.close()

        valid_agents_per_scene = self.target_vehicle_mask.sum(dim=1).float()
        print("Combined dataset loaded:")
        print(f"  path: {combined_path}")
        print(f"  mode: {self.train_mode}")
        print(f"  label source: {self.label_source}")
        print(f"  samples: {len(self.labels)}")
        print(f"  history shape: {tuple(self.history_data.shape)}")
        print(f"  target shape: {tuple(self.target_data.shape)}")
        print(f"  map shape: {tuple(self.map_data.shape)}")
        print(f"  avg valid agents: {valid_agents_per_scene.mean().item():.2f}")
        if self.train_mode == "prediction":
            print(
                "  dropped future-only support: "
                f"agents={self.dropped_future_only_agents}, "
                f"timesteps={self.dropped_future_only_points}"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.history_data[idx],
            self.target_data[idx],
            self.labels[idx],
            self.map_data[idx],
            self.map_mask[idx],
            self.agent_types[idx],
            self.scene_stats[idx],
            self.map_names[idx],
            self.target_vehicle_mask[idx],
            self.history_vehicle_mask[idx],
            self.map_type[idx],
            self.map_speed_limit[idx],
            self.context_agent_mask[idx],
            self.loss_timestep_mask[idx],
            self.history_timestep_mask[idx],
            self.static_dimensions[idx],
            self.state_scales[idx],
            self.map_lanelet_id[idx],
            self.map_lane_subtype[idx],
            self.map_left_boundary_type[idx],
            self.map_right_boundary_type[idx],
            self.lane_edge_index[idx],
            self.lane_edge_type[idx],
            self.lane_edge_mask[idx],
        )


def calc_z_shapes(n_channel, input_size_h, input_size_w, n_block):
    divisor = 2 ** int(n_block)
    if input_size_h % divisor != 0:
        raise ValueError(
            f"input_size_h={input_size_h} must be divisible by 2**n_block={divisor}. "
            "Choose prediction_target_steps and n_block so no model padding is required."
        )
    z_shapes = []
    for _ in range(n_block - 1):
        input_size_h //= 2
        z_shapes.append((n_channel, input_size_h, input_size_w))

    input_size_h //= 2
    z_shapes.append((n_channel * 2, input_size_h, input_size_w))
    return z_shapes


def sample_latents(batch_size, z_shapes, device, base_temp, block_decay=1.0):
    temps = [base_temp * (block_decay ** idx) for idx in range(len(z_shapes))]
    return [torch.randn(batch_size, *z, device=device) * temps[idx] for idx, z in enumerate(z_shapes)]


def crop_time_steps(array, steps, axis):
    """Return exactly the requested leading timesteps along ``axis``."""
    steps = int(steps)
    axis = int(axis) % array.ndim
    available = int(array.shape[axis])
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if available < steps:
        raise ValueError(f"cannot crop {steps} timesteps from axis length {available}")
    slices = [slice(None)] * array.ndim
    slices[axis] = slice(0, steps)
    return array[tuple(slices)]


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def build_map_features(map_xy, map_mask, map_type):
    """
    map_xy:   [B, L, P, 2]
    map_mask: [B, L, P]
    map_type: [B, L]
    return:   [B, L, P, 6] = [x, y, yaw/pi, type0, type1, type2]
    """
    B, L, P, _ = map_xy.shape
    yaw = torch.zeros(B, L, P, 1, device=map_xy.device, dtype=map_xy.dtype)
    if P > 1:
        diff = map_xy[:, :, 1:, :] - map_xy[:, :, :-1, :]
        yaw_vals = torch.atan2(diff[..., 1], diff[..., 0])
        yaw[:, :, :-1, 0] = yaw_vals
        yaw[:, :, -1, 0] = yaw[:, :, -2, 0]
    yaw = yaw / torch.pi

    type_ids = map_type.long().clamp(min=0, max=2)
    type_onehot = torch.nn.functional.one_hot(type_ids, num_classes=3).to(dtype=map_xy.dtype)
    type_onehot = type_onehot.unsqueeze(2).expand(-1, -1, P, -1)
    type_onehot = type_onehot * map_mask.unsqueeze(-1).float()
    return torch.cat([map_xy, yaw, type_onehot], dim=-1)


def load_yaml_config(config_path, parser):
    if not config_path:
        return {}

    with open(config_path, "r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}

    if not isinstance(config_data, dict):
        raise ValueError(f"config file must contain a dict: {config_path}")

    valid_keys = {
        action.dest for action in parser._actions
        if action.dest not in (argparse.SUPPRESS, "help")
    }
    unknown_keys = sorted(set(config_data.keys()) - valid_keys)
    if unknown_keys:
        raise ValueError(
            f"config file contains unknown args: {unknown_keys}. "
            "Please keep YAML keys aligned with argparse names."
        )
    return config_data


def safe_load_state(path, map_location="cpu"):
    if path is None:
        return None
    if not os.path.exists(path):
        print(f"[safe_load_state] missing file: {path}")
        return None
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        print(f"[safe_load_state] failed to load {path}: {exc}")
        return None


def build_dataloader(
    args,
    is_distributed,
    rank,
    world_size,
    combined_path=None,
    shuffle=True,
    batch_size=None,
):
    dataset = CombinedInteractionDataset(
        combined_path=combined_path or args.combined_path,
        in_channel=args.in_channel,
        train_mode=args.train_mode,
        history_steps=args.history_steps,
        future_steps=args.future_steps,
        prediction_target_steps=args.prediction_target_steps,
        prediction_representation=getattr(args, "prediction_representation", "absolute"),
        label_source=args.label_source,
        turn_angle_threshold_deg=args.turn_angle_threshold_deg,
        stationary_dist_threshold=args.stationary_dist_threshold,
        allow_legacy_prediction_data=getattr(args, "allow_legacy_prediction_data", False),
        require_map_topology=getattr(args, "model_arch", "mapglow") == "route_mixture_glow",
    )

    dataloader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": bool(is_distributed and shuffle),
    }
    if args.num_workers > 0:
        if args.worker_start_method:
            dataloader_kwargs["multiprocessing_context"] = args.worker_start_method
        dataloader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        if args.prefetch_factor > 0:
            dataloader_kwargs["prefetch_factor"] = args.prefetch_factor

    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=bool(shuffle),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size or args.batch,
            sampler=sampler,
            **dataloader_kwargs,
        )
    else:
        sampler = None
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size or args.batch,
            shuffle=shuffle,
            **dataloader_kwargs,
        )

    return dataloader, sampler, dataset


def build_random_validation_sample_dataloader(dataset, args):
    """Build a rank-local shuffled validation loader for periodic inference.

    Only the main rank consumes this loader. It shares the already-loaded
    validation dataset and uses a dedicated RNG, so choosing visualization
    scenes neither reads the training loader nor perturbs training shuffling.
    """
    generator = torch.Generator()
    sample_seed = int(getattr(args, "sample_seed", 12345))
    if sample_seed >= 0:
        generator.manual_seed(sample_seed + 1)
    else:
        generator.seed()
    return DataLoader(
        dataset,
        batch_size=args.val_batch if args.val_batch > 0 else args.batch,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def next_validation_sample_batch(
    fixed_batch, random_dataloader, random_iterator
):
    """Return a validation batch and the reusable random-loader iterator."""
    if fixed_batch is not None:
        return fixed_batch, random_iterator
    if random_dataloader is None:
        raise RuntimeError(
            "periodic inference requires a validation dataset via val_combined_path; "
            "training batches are never used for sampling"
        )
    if random_iterator is None:
        random_iterator = iter(random_dataloader)
    try:
        batch = next(random_iterator)
    except StopIteration:
        random_iterator = iter(random_dataloader)
        batch = next(random_iterator)
    return batch, random_iterator


def process_batch(batch_raw, device, train_mode="initialization"):
    use_history = train_mode == "prediction"
    history_data = batch_raw[0].to(device, non_blocking=True) if use_history else None
    target_data = batch_raw[1].to(device, non_blocking=True)
    labels = batch_raw[2].to(device, non_blocking=True)
    map_xy = batch_raw[3].to(device, non_blocking=True)
    map_mask = batch_raw[4].to(device, non_blocking=True)
    agent_types = batch_raw[5].to(device, non_blocking=True)
    scene_stats = batch_raw[6].to(device, non_blocking=True)
    map_name = batch_raw[7]
    target_vehicle_mask = batch_raw[8].to(device, non_blocking=True)
    history_vehicle_mask = batch_raw[9].to(device, non_blocking=True) if use_history else None
    map_type = batch_raw[10].to(device, non_blocking=True)
    map_speed_limit = batch_raw[11].to(device, non_blocking=True)
    context_agent_mask = (
        batch_raw[12].to(device, non_blocking=True)
        if len(batch_raw) > 12
        else history_vehicle_mask if use_history else target_vehicle_mask
    )
    loss_timestep_mask = (
        batch_raw[13].to(device, non_blocking=True)
        if len(batch_raw) > 13
        else ~torch.isclose(target_data, torch.tensor(-1.0, device=device), atol=0.05).all(dim=1)
    )
    history_timestep_mask = (
        batch_raw[14].to(device, non_blocking=True)
        if len(batch_raw) > 14 and use_history
        else None
    )
    static_dimensions = (
        batch_raw[15].to(device, non_blocking=True)
        if len(batch_raw) > 15
        else None
    )
    state_scales = (
        batch_raw[16].to(device, non_blocking=True)
        if len(batch_raw) > 16
        else torch.stack(
            [
                scene_stats[:, 2].abs().clamp_min(1e-6),
                torch.full_like(scene_stats[:, 2], 15.0),
                torch.full_like(scene_stats[:, 2], float(np.pi)),
            ],
            dim=1,
        )
    )
    map_lanelet_id = batch_raw[17].to(device, non_blocking=True) if len(batch_raw) > 17 else None
    map_lane_subtype = batch_raw[18].to(device, non_blocking=True) if len(batch_raw) > 18 else None
    map_left_boundary_type = batch_raw[19].to(device, non_blocking=True) if len(batch_raw) > 19 else None
    map_right_boundary_type = batch_raw[20].to(device, non_blocking=True) if len(batch_raw) > 20 else None
    lane_edge_index = batch_raw[21].to(device, non_blocking=True) if len(batch_raw) > 21 else None
    lane_edge_type = batch_raw[22].to(device, non_blocking=True) if len(batch_raw) > 22 else None
    lane_edge_mask = batch_raw[23].to(device, non_blocking=True) if len(batch_raw) > 23 else None
    raw_labels = labels.clone()
    raw_agent_types = agent_types.clone()

    map_data = build_map_features(map_xy, map_mask, map_type)
    timestep_mask = loss_timestep_mask.bool()  # compatibility alias

    valid_mask = context_agent_mask.bool()
    labels = torch.where(valid_mask, labels + 1, torch.zeros_like(labels))
    agent_types = torch.where(valid_mask, agent_types + 1, torch.zeros_like(agent_types))

    return {
        "history_data": history_data,
        "target_data": target_data,
        "labels": labels,
        "raw_labels": raw_labels,
        "map_data": map_data,
        "map_mask": map_mask,
        "agent_types": agent_types,
        "raw_agent_types": raw_agent_types,
        "scene_stats": scene_stats,
        "map_name": map_name,
        "target_vehicle_mask": target_vehicle_mask,
        "context_agent_mask": context_agent_mask.bool(),
        "history_vehicle_mask": history_vehicle_mask,
        "timestep_mask": timestep_mask,
        "loss_timestep_mask": loss_timestep_mask.bool(),
        "history_timestep_mask": history_timestep_mask.bool() if history_timestep_mask is not None else None,
        "static_dimensions": static_dimensions,
        "state_scales": state_scales,
        "map_type": map_type,
        "map_speed_limit": map_speed_limit,
        "map_lanelet_id": map_lanelet_id,
        "map_lane_subtype": map_lane_subtype,
        "map_left_boundary_type": map_left_boundary_type,
        "map_right_boundary_type": map_right_boundary_type,
        "lane_edge_index": lane_edge_index,
        "lane_edge_type": lane_edge_type,
        "lane_edge_mask": lane_edge_mask,
    }


def decode_prediction_states(states, batch, args, timestep_mask=None):
    representation = getattr(args, "prediction_representation", "absolute")
    if args.train_mode != "prediction" or representation == "absolute":
        return states
    history_timestep_mask = batch.get("history_timestep_mask")
    if history_timestep_mask is None:
        raise ValueError(
            f"{representation} prediction requires an explicit history_timestep_mask"
        )
    decode_mask = batch["timestep_mask"] if timestep_mask is None else timestep_mask
    if representation == "delta":
        return decode_state_deltas_torch(
            states,
            batch["history_data"],
            decode_mask,
            history_timestep_mask,
            yaw_scale=batch["state_scales"][:, 2],
        )
    if representation == "cv_residual":
        return decode_constant_velocity_residual_torch(
            states,
            batch["history_data"],
            decode_mask,
            history_timestep_mask,
            yaw_scale=batch["state_scales"][:, 2],
        )
    raise ValueError(f"unsupported prediction representation: {representation}")


def _method_keyword_names(model, method_name):
    method = getattr(unwrap_model(model), method_name)
    signature = inspect.signature(method)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    return set(signature.parameters), accepts_kwargs


def filter_model_kwargs(model, model_kwargs, method_name="forward"):
    names, accepts_kwargs = _method_keyword_names(model, method_name)
    if accepts_kwargs:
        return {key: value for key, value in model_kwargs.items() if value is not None}
    return {
        key: value
        for key, value in model_kwargs.items()
        if value is not None and key in names
    }


def build_generation_timestep_mask(batch, train_mode="initialization", future_steps=None):
    if train_mode != "prediction":
        return batch.get("loss_timestep_mask", batch["timestep_mask"])
    target_steps = int(batch["target_data"].shape[2])
    real_steps = target_steps if future_steps is None else min(int(future_steps), target_steps)
    mask = torch.zeros(
        batch["context_agent_mask"].shape[0],
        target_steps,
        batch["context_agent_mask"].shape[1],
        dtype=torch.bool,
        device=batch["context_agent_mask"].device,
    )
    mask[:, :real_steps, :] = batch["context_agent_mask"].unsqueeze(1)
    return mask


def build_model_context_kwargs(
    batch,
    train_mode="initialization",
    model=None,
    method_name="forward",
    for_sampling=False,
    future_steps=None,
):
    use_history = train_mode == "prediction"
    loss_mask = batch.get("loss_timestep_mask", batch["timestep_mask"])
    flow_mask = (
        build_generation_timestep_mask(batch, train_mode, future_steps=future_steps)
        if for_sampling
        else loss_mask
    )
    model_kwargs = {
        "map_data": batch["map_data"],
        "map_mask": batch["map_mask"],
        "agent_types": batch["agent_types"],
        # Existing models call this target_vehicle_mask, but it is conditioner-only.
        "target_vehicle_mask": batch["context_agent_mask"],
        "context_agent_mask": batch["context_agent_mask"],
        "timestep_mask": flow_mask,
        "loss_timestep_mask": loss_mask if not for_sampling else None,
        "scene_stats": batch.get("scene_stats"),
        "static_dimensions": batch.get("static_dimensions"),
        "state_scales": batch.get("state_scales"),
        "map_speed_limit": batch.get("map_speed_limit"),
        "map_lanelet_id": batch.get("map_lanelet_id"),
        "map_lane_subtype": batch.get("map_lane_subtype"),
        "map_left_boundary_type": batch.get("map_left_boundary_type"),
        "map_right_boundary_type": batch.get("map_right_boundary_type"),
        "lane_edge_index": batch.get("lane_edge_index"),
        "lane_edge_type": batch.get("lane_edge_type"),
        "lane_edge_mask": batch.get("lane_edge_mask"),
    }
    if use_history:
        model_kwargs["history_data"] = batch["history_data"]
        model_kwargs["history_vehicle_mask"] = batch["history_vehicle_mask"]
        model_kwargs["history_timestep_mask"] = batch.get("history_timestep_mask")
    return (
        filter_model_kwargs(model, model_kwargs, method_name=method_name)
        if model is not None
        else {key: value for key, value in model_kwargs.items() if value is not None}
    )


def initialize_model_once(model, dataloader, device, is_distributed, local_rank, sampler=None, train_mode="initialization"):
    if sampler is not None:
        sampler.set_epoch(0)

    if (not is_distributed) or local_rank == 0:
        warmup_iter = iter(dataloader)
        batch = process_batch(next(warmup_iter), device, train_mode=train_mode)

        with torch.no_grad():
            dataset_label_source = getattr(dataloader.dataset, "label_source", "auto")
            condition = batch["labels"] if uses_label_condition(train_mode, dataset_label_source) else None
            _ = model(
                batch["target_data"],
                condition=condition,
                **build_model_context_kwargs(batch, train_mode=train_mode, model=model),
            )

    if is_distributed:
        dist.barrier()
        for param in model.parameters():
            broadcast_tensor_from_rank0_(param.data)
        for buffer in model.buffers():
            broadcast_tensor_from_rank0_(buffer.data)
        dist.barrier()


@contextlib.contextmanager
def evaluating(model):
    was_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        if was_training is not None and hasattr(model, "train"):
            model.train(was_training)


@contextlib.contextmanager
def preserving_rng_state():
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


def save_samples_npz(model_single, batch, args, z_shapes, device, step):
    batch_size = min(args.batch, batch["target_data"].shape[0])
    sample_labels = batch["labels"][:batch_size]
    sample_raw_labels = batch["raw_labels"][:batch_size]
    sample_scene_stats = batch["scene_stats"][:batch_size]
    sample_state_scales = batch.get("state_scales")
    if sample_state_scales is None:
        sample_state_scales = torch.stack(
            [
                sample_scene_stats[:, 2].abs().clamp_min(1e-6),
                torch.full_like(sample_scene_stats[:, 2], 15.0),
                torch.full_like(sample_scene_stats[:, 2], float(np.pi)),
            ],
            dim=1,
        )
    else:
        sample_state_scales = sample_state_scales[:batch_size]
    sample_map_name = batch["map_name"][:batch_size] if hasattr(batch["map_name"], "__len__") else batch["map_name"]
    sample_gt = batch["target_data"][:batch_size]
    sample_map_type = batch["map_type"][:batch_size]
    sample_map_speed_limit = batch["map_speed_limit"][:batch_size]
    sample_batch = {
        "target_data": batch["target_data"][:batch_size],
        "map_data": batch["map_data"][:batch_size],
        "map_mask": batch["map_mask"][:batch_size],
        "agent_types": batch["agent_types"][:batch_size],
        "raw_agent_types": batch["raw_agent_types"][:batch_size],
        "target_vehicle_mask": batch["target_vehicle_mask"][:batch_size],
        "context_agent_mask": batch.get("context_agent_mask", batch["target_vehicle_mask"])[:batch_size],
        "timestep_mask": batch["timestep_mask"][:batch_size],
        "loss_timestep_mask": batch.get("loss_timestep_mask", batch["timestep_mask"])[:batch_size],
        "scene_stats": batch["scene_stats"][:batch_size],
        "state_scales": sample_state_scales,
        "static_dimensions": (
            batch["static_dimensions"][:batch_size]
            if batch.get("static_dimensions") is not None else None
        ),
        "history_data": (
            batch["history_data"][:batch_size]
            if args.train_mode == "prediction" and batch["history_data"] is not None
            else None
        ),
        "history_vehicle_mask": (
            batch["history_vehicle_mask"][:batch_size]
            if args.train_mode == "prediction" and batch["history_vehicle_mask"] is not None
            else None
        ),
        "history_timestep_mask": (
            batch["history_timestep_mask"][:batch_size]
            if args.train_mode == "prediction" and batch.get("history_timestep_mask") is not None
            else None
        ),
        "map_speed_limit": batch.get("map_speed_limit")[:batch_size] if batch.get("map_speed_limit") is not None else None,
        "map_lanelet_id": batch.get("map_lanelet_id")[:batch_size] if batch.get("map_lanelet_id") is not None else None,
        "map_lane_subtype": batch.get("map_lane_subtype")[:batch_size] if batch.get("map_lane_subtype") is not None else None,
        "map_left_boundary_type": batch.get("map_left_boundary_type")[:batch_size] if batch.get("map_left_boundary_type") is not None else None,
        "map_right_boundary_type": batch.get("map_right_boundary_type")[:batch_size] if batch.get("map_right_boundary_type") is not None else None,
        "lane_edge_index": batch.get("lane_edge_index")[:batch_size] if batch.get("lane_edge_index") is not None else None,
        "lane_edge_type": batch.get("lane_edge_type")[:batch_size] if batch.get("lane_edge_type") is not None else None,
        "lane_edge_mask": batch.get("lane_edge_mask")[:batch_size] if batch.get("lane_edge_mask") is not None else None,
    }
    sample_model_kwargs = build_model_context_kwargs(
        sample_batch,
        train_mode=args.train_mode,
        model=model_single,
        method_name="reverse",
        for_sampling=True,
        future_steps=args.future_steps,
    )
    generation_timestep_mask = build_generation_timestep_mask(
        sample_batch,
        train_mode=args.train_mode,
        future_steps=args.future_steps,
    )
    label_condition_used = uses_label_condition(args.train_mode, args.label_source)

    conditional_samples = []
    unconditional_samples = []

    with preserving_rng_state(), evaluating(model_single):
        sample_seed = int(getattr(args, "sample_seed", 12345))
        if sample_seed >= 0:
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)
        for mode_sample_index in range(args.n_modes):
            z_cond = sample_latents(batch_size, z_shapes, device, args.temp, args.temp_block_decay)
            route_reverse_kwargs = {}
            if getattr(args, "model_arch", "mapglow") == "route_mixture_glow":
                route_reverse_kwargs["mode_index"] = mode_sample_index % args.route_modes
            if label_condition_used:
                cond_sample = model_single.reverse(
                    z_cond,
                    sample_labels,
                    guidance_scale=args.cfg_scale,
                    **route_reverse_kwargs,
                    **sample_model_kwargs,
                ).detach()
            else:
                cond_sample = model_single.reverse(
                    z_cond,
                    **route_reverse_kwargs,
                    **sample_model_kwargs,
                ).detach()
            cond_sample = decode_prediction_states(
                cond_sample,
                sample_batch,
                args,
                timestep_mask=generation_timestep_mask,
            ).cpu()
            conditional_samples.append(cond_sample.numpy())

            z_uncond = sample_latents(batch_size, z_shapes, device, args.temp, args.temp_block_decay)
            uncond_sample = model_single.reverse(
                z_uncond,
                **route_reverse_kwargs,
                **sample_model_kwargs,
            ).detach()
            uncond_sample = decode_prediction_states(
                uncond_sample,
                sample_batch,
                args,
                timestep_mask=generation_timestep_mask,
            ).cpu()
            unconditional_samples.append(uncond_sample.numpy())

    conditional_samples = np.stack(conditional_samples, axis=0)
    unconditional_samples = np.stack(unconditional_samples, axis=0)
    route_save_payload = {}
    if getattr(args, "model_arch", "mapglow") == "route_mixture_glow":
        route_outputs = model_single._route_outputs(**sample_model_kwargs)
        route_save_payload = {
            "route_probabilities": route_outputs["route_probabilities"].detach().cpu().numpy(),
            "route_logits": route_outputs["route_logits"].detach().cpu().numpy(),
            "route_proposals_cv_residual": route_outputs["proposal"].detach().cpu().numpy(),
        }
    saved_gt_tensor = decode_prediction_states(sample_gt, sample_batch, args)
    saved_gt = np.array(saved_gt_tensor.cpu().data)
    saved_timestep_mask = np.array(sample_batch["timestep_mask"].cpu().data)
    if args.train_mode == "prediction":
        conditional_samples = crop_time_steps(conditional_samples, args.future_steps, axis=3)
        unconditional_samples = crop_time_steps(unconditional_samples, args.future_steps, axis=3)
        saved_gt = crop_time_steps(saved_gt, args.future_steps, axis=2)
        saved_timestep_mask = crop_time_steps(saved_timestep_mask, args.future_steps, axis=1)

    out_dir = Path(args.sample_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{step + 1:06d}_interaction_combined_samples.npz"
    save_dict = dict(
        conditional_samples=conditional_samples,
        unconditional_samples=unconditional_samples,
        gt=saved_gt,
        labels=np.array(sample_raw_labels.cpu().data),
        model_labels=np.array(sample_labels.cpu().data),
        maps=np.array(sample_batch["map_data"].cpu().data),
        map_mask=np.array(sample_batch["map_mask"].cpu().data),
        map_type=np.array(sample_map_type.cpu().data),
        map_speed_limit=np.array(sample_map_speed_limit.cpu().data),
        agent_types=np.array(sample_batch["raw_agent_types"].cpu().data),
        model_agent_types=np.array(sample_batch["agent_types"].cpu().data),
        scene_stats=np.array(sample_scene_stats.cpu().data),
        state_scales=np.array(sample_batch["state_scales"].cpu().data),
        map_name=np.array(sample_map_name),
        target_vehicle_mask=np.array(sample_batch["target_vehicle_mask"].cpu().data),
        context_agent_mask=np.array(sample_batch["context_agent_mask"].cpu().data),
        timestep_mask=saved_timestep_mask,
        n_modes=args.n_modes,
        in_channel=args.in_channel,
        train_mode=np.asarray(args.train_mode),
        use_history=np.bool_(args.train_mode == "prediction"),
        label_condition_used=np.bool_(label_condition_used),
        history_steps=np.int64(args.history_steps),
        future_steps=np.int64(args.future_steps),
        prediction_target_steps=np.int64(args.prediction_target_steps),
        saved_target_steps=np.int64(conditional_samples.shape[3]),
        label_source=np.asarray(args.label_source),
        trajectory_representation=np.asarray("absolute"),
        model_output_representation=np.asarray(
            getattr(args, "prediction_representation", "absolute")
        ),
        model_arch=np.asarray(getattr(args, "model_arch", "mapglow")),
        **route_save_payload,
    )
    if args.train_mode == "prediction":
        save_dict["history_data"] = np.array(sample_batch["history_data"].cpu().data)
        save_dict["history_vehicle_mask"] = np.array(sample_batch["history_vehicle_mask"].cpu().data)
        if sample_batch["history_timestep_mask"] is not None:
            save_dict["history_timestep_mask"] = np.array(
                sample_batch["history_timestep_mask"].cpu().data
            )
        if sample_batch["static_dimensions"] is not None:
            save_dict["static_dimensions"] = np.array(sample_batch["static_dimensions"].cpu().data)
    np.savez(out_path, **save_dict)
    print(f"[samples] saved to {out_path}")

    if args.save_sample_images and args.train_mode == "prediction":
        save_sample_visualizations(
            out_dir=out_dir,
            step=step,
            prediction_samples=unconditional_samples,
            gt=saved_gt,
            history_data=(
                np.array(sample_batch["history_data"].cpu().data)
                if args.train_mode == "prediction" and sample_batch["history_data"] is not None
                else None
            ),
            map_data=np.array(sample_batch["map_data"].cpu().data),
            map_mask=np.array(sample_batch["map_mask"].cpu().data),
            map_type=np.array(sample_map_type.cpu().data),
            map_name=np.array(sample_map_name),
            target_vehicle_mask=np.array(sample_batch["target_vehicle_mask"].cpu().data),
            timestep_mask=saved_timestep_mask,
            args=args,
        )


MAP_STYLE = {
    0: {"color": "#111827", "lw": 1.25, "alpha": 0.95, "label": "map centerline"},
    1: {"color": "#9ca3af", "lw": 0.9, "alpha": 0.75, "label": "map boundary"},
    2: {"color": "#f59e0b", "lw": 1.0, "alpha": 0.75, "label": "crosswalk"},
}


def vis_traj_xy(traj):
    return traj[:2].copy()


def vis_infer_map_type(map_data, map_type, lane_idx, valid_len):
    if map_type is not None and lane_idx < len(map_type):
        return int(np.clip(map_type[lane_idx], 0, 2))
    if valid_len <= 0 or map_data.shape[-1] < 6:
        return -1
    type_scores = np.mean(map_data[lane_idx, :valid_len, 3:6], axis=0)
    return int(np.argmax(type_scores))


def vis_valid_path(traj_xy, agent_idx, timestep_mask=None):
    path = traj_xy[:, :, agent_idx].T
    if timestep_mask is not None:
        valid_t = timestep_mask[:, agent_idx]
        if not valid_t.any():
            return None
        path = path[valid_t]
    else:
        valid_t = ~(
            np.isclose(path[:, 0], -1.0, atol=0.05)
            & np.isclose(path[:, 1], -1.0, atol=0.05)
        )
        path = path[valid_t]
        if len(path) == 0:
            return None
    path = path[np.isfinite(path).all(axis=1)]
    return path if len(path) > 0 else None


def vis_compute_limits(map_xy, map_mask, limit_paths, pad):
    points = []

    for lane_idx in range(map_xy.shape[0]):
        valid_len = int(map_mask[lane_idx].sum())
        if valid_len > 1:
            points.append(map_xy[lane_idx, :valid_len, :2])

    points.extend(path[:, :2] for path in limit_paths if path is not None and len(path) > 0)
    points.append(np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32))

    if not points:
        return (-1.1, 1.1), (-1.1, 1.1)

    pts = np.concatenate(points, axis=0)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return (-1.1, 1.1), (-1.1, 1.1)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    span = np.maximum(np.array([x_max - x_min, y_max - y_min]), 0.25)
    pad_ratio = float(pad)
    if pad_ratio < 0.0 or pad_ratio >= 1.0:
        pad_ratio = 0.08
    pad_xy = span * pad_ratio
    return (float(x_min - pad_xy[0]), float(x_max + pad_xy[0])), (
        float(y_min - pad_xy[1]),
        float(y_max + pad_xy[1]),
    )


def vis_draw_map(ax, map_data, map_xy, map_mask, map_type):
    for lane_idx in range(map_xy.shape[0]):
        valid_len = int(map_mask[lane_idx].sum())
        if valid_len <= 1:
            continue
        lane = map_xy[lane_idx, :valid_len, :2]
        lane_type = vis_infer_map_type(map_data, map_type, lane_idx, valid_len)
        style = MAP_STYLE.get(lane_type, {"color": "#9ca3af", "lw": 0.8, "alpha": 0.6, "label": "map"})
        ax.plot(
            lane[:, 0],
            lane[:, 1],
            color=style["color"],
            alpha=style["alpha"],
            linewidth=style["lw"],
            zorder=1,
            solid_capstyle="round",
        )


def vis_draw_paths(ax, traj_xy, agent_indices, timestep_mask, color, label, linestyle="-", linewidth=2.0, alpha=0.95):
    first_line = None
    for agent_idx in agent_indices:
        path = vis_valid_path(traj_xy, agent_idx, timestep_mask)
        if path is None or path.shape[0] < 2:
            continue
        line = ax.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            zorder=4,
            solid_capstyle="round",
        )[0]
        first_line = first_line or line
        ax.scatter(
            path[0, 0],
            path[0, 1],
            s=16,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            s=26,
            marker="x",
            color=color,
            linewidth=1.2,
            zorder=5,
        )
    if first_line is not None:
        first_line.set_label(label)
    return first_line


def vis_history_mask(history_data):
    if history_data is None:
        return None
    return ~np.isclose(history_data[:, :5], -1.0, atol=0.05).all(axis=1)


def vis_setup_axis(ax, map_data, map_xy, map_mask, map_type, xlim, ylim):
    from matplotlib.patches import Rectangle

    ax.set_facecolor("#ffffff")
    vis_draw_map(ax, map_data, map_xy, map_mask, map_type)
    ax.add_patch(
        Rectangle(
            (-1.0, -1.0),
            2.0,
            2.0,
            fill=False,
            linestyle="--",
            linewidth=1.0,
            edgecolor="#ef4444",
            alpha=0.75,
            zorder=2,
        )
    )
    ax.axhline(0.0, color="#d1d5db", linewidth=0.7, zorder=0)
    ax.axvline(0.0, color="#d1d5db", linewidth=0.7, zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("normalized x")
    ax.set_ylabel("normalized y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e5e7eb", linewidth=0.6)


def save_single_visualization(image_path, title, pred_xy, gt_xy, history_xy, map_data, map_mask, map_type, map_name, target_vehicle_mask, timestep_mask, history_timestep_mask, args):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    map_xy = map_data[..., :2].copy()
    pred_xy = vis_traj_xy(pred_xy)
    gt_xy = vis_traj_xy(gt_xy) if gt_xy is not None else None
    history_xy = vis_traj_xy(history_xy) if history_xy is not None else None

    agent_indices = np.where(target_vehicle_mask.astype(bool))[0]
    limit_paths = []
    for agent_idx in agent_indices:
        if history_xy is not None:
            limit_paths.append(vis_valid_path(history_xy, agent_idx, history_timestep_mask))
        if gt_xy is not None:
            limit_paths.append(vis_valid_path(gt_xy, agent_idx, timestep_mask))
    xlim, ylim = vis_compute_limits(map_xy, map_mask, limit_paths, args.vis_pad)

    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    fig.patch.set_facecolor("#ffffff")

    vis_setup_axis(ax, map_data, map_xy, map_mask, map_type, xlim, ylim)
    handles = [
        Line2D([0], [0], color=style["color"], linewidth=style["lw"], alpha=style["alpha"], label=style["label"])
        for style in MAP_STYLE.values()
    ] + [
        Line2D([0], [0], color="#ef4444", linestyle="--", linewidth=1.0, label="[-1, 1] box"),
    ]

    if history_xy is not None:
        handle = vis_draw_paths(
            ax,
            history_xy,
            agent_indices,
            history_timestep_mask,
            color="#64748b",
            label="history",
            linestyle="-",
            linewidth=1.8,
            alpha=0.9,
        )
        if handle is not None:
            handles.append(Line2D([0], [0], color="#64748b", linewidth=1.8, label="history"))
    if gt_xy is not None:
        handle = vis_draw_paths(
            ax,
            gt_xy,
            agent_indices,
            timestep_mask,
            color="#2563eb",
            label="gt future",
            linestyle="-",
            linewidth=2.2,
            alpha=0.95,
        )
        if handle is not None:
            handles.append(Line2D([0], [0], color="#2563eb", linewidth=2.2, label="gt future"))
    if pred_xy is not None:
        handle = vis_draw_paths(
            ax,
            pred_xy,
            agent_indices,
            timestep_mask,
            color="#ea580c",
            label="prediction",
            linestyle="-",
            linewidth=2.2,
            alpha=0.95,
        )
        if handle is not None:
            handles.append(Line2D([0], [0], color="#ea580c", linewidth=2.2, label="prediction"))
    ax.set_title(f"{title} | {map_name}")
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)

    fig.tight_layout()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=args.vis_dpi)
    plt.close(fig)


def save_sample_visualizations(
    out_dir,
    step,
    prediction_samples,
    gt,
    history_data,
    map_data,
    map_mask,
    map_type,
    map_name,
    target_vehicle_mask,
    timestep_mask,
    args,
):
    image_dir = out_dir / f"{step + 1:06d}_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    num_scenes = min(args.vis_num_scenes, gt.shape[0])
    num_modes = min(args.vis_num_modes, prediction_samples.shape[0])

    for scene_idx in range(num_scenes):
        scene_name = str(map_name[scene_idx]) if np.ndim(map_name) > 0 else str(map_name)
        for mode_idx in range(num_modes):
            image_path = image_dir / f"scene{scene_idx:03d}_mode{mode_idx:02d}_prediction.png"

            save_single_visualization(
                image_path=image_path,
                title=f"Prediction | mode {mode_idx}",
                pred_xy=prediction_samples[mode_idx, scene_idx],
                gt_xy=gt[scene_idx],
                history_xy=history_data[scene_idx] if history_data is not None else None,
                map_data=map_data[scene_idx],
                map_mask=map_mask[scene_idx],
                map_type=map_type[scene_idx] if map_type is not None else None,
                map_name=scene_name,
                target_vehicle_mask=target_vehicle_mask[scene_idx],
                timestep_mask=timestep_mask[scene_idx],
                history_timestep_mask=vis_history_mask(history_data[scene_idx:scene_idx + 1])[0] if history_data is not None else None,
                args=args,
            )

    print(f"[samples] saved images to {image_dir}")


def trajectory_metric_sums(
    samples,
    target,
    timestep_mask,
    agent_mask,
    position_scale,
    *,
    history=None,
    history_timestep_mask=None,
    velocity_scale=None,
    yaw_scale=None,
    turn_angle_threshold_deg=30.0,
    timestep_seconds=0.1,
):
    """Return trajectory, turning, and kinematic diagnostic sums."""
    if samples.ndim != 5 or target.ndim != 4:
        raise ValueError("samples/target must be [K,B,C,T,V] and [B,C,T,V]")
    steps = min(samples.shape[3], target.shape[2], timestep_mask.shape[1])
    full_samples = samples[:, :, :, :steps]
    full_target = target[:, :, :steps]
    sample_xy = full_samples[:, :, :2]
    target_xy = full_target[:, :2]
    mask = timestep_mask[:, :steps].bool()
    valid_agent = mask.any(dim=1) & agent_mask.bool()

    scale = position_scale.to(sample_xy).reshape(1, -1, 1, 1)
    error = torch.linalg.vector_norm(
        sample_xy - target_xy.unsqueeze(0), dim=2
    ) * scale
    masked_error = error * mask.unsqueeze(0)
    counts = mask.sum(dim=1).clamp_min(1).unsqueeze(0)
    ade = masked_error.sum(dim=2) / counts

    time_index = torch.arange(steps, device=mask.device).view(1, steps, 1)
    last_index = torch.where(mask, time_index, -1).amax(dim=1).clamp_min(0)
    fde = error.gather(
        dim=2,
        index=last_index.unsqueeze(0).expand(error.shape[0], -1, -1).unsqueeze(2),
    ).squeeze(2)
    valid = valid_agent.to(error.dtype)
    result = {
        "agent_count": valid.sum(),
        "ade_sum": (ade[0] * valid).sum(),
        "fde_sum": (fde[0] * valid).sum(),
        "minade_sum": (ade.min(dim=0).values * valid).sum(),
        "minfde_sum": (fde.min(dim=0).values * valid).sum(),
    }

    if (
        history is not None
        and history_timestep_mask is not None
        and full_target.shape[1] > 4
    ):
        history_mask = history_timestep_mask.bool()
        hist_time = torch.arange(history.shape[2], device=history.device).view(1, -1, 1)
        last_hist_idx = torch.where(history_mask, hist_time, -1).amax(dim=1).clamp_min(0)
        history_yaw = torch.gather(
            history[:, 4], 1, last_hist_idx.unsqueeze(1)
        ).squeeze(1)
        target_yaw = torch.gather(
            full_target[:, 4], 1, last_index.unsqueeze(1)
        ).squeeze(1)
        sample_yaw = torch.gather(
            full_samples[:, :, 4],
            2,
            last_index.unsqueeze(0).unsqueeze(2).expand(
                full_samples.shape[0], -1, 1, -1
            ),
        ).squeeze(2)
        yaw_scale_tensor = (
            yaw_scale.to(target_yaw).view(-1, 1)
            if yaw_scale is not None
            else torch.full(
                (target_yaw.shape[0], 1), torch.pi,
                dtype=target_yaw.dtype, device=target_yaw.device,
            )
        )
        target_turn = torch.abs(
            torch.remainder(
                (target_yaw - history_yaw) * yaw_scale_tensor + torch.pi,
                2 * torch.pi,
            ) - torch.pi
        )
        sample_turn = torch.abs(
            torch.remainder(
                (sample_yaw - history_yaw.unsqueeze(0))
                * yaw_scale_tensor.unsqueeze(0) + torch.pi,
                2 * torch.pi,
            ) - torch.pi
        )
        threshold = torch.deg2rad(
            target_turn.new_tensor(float(turn_angle_threshold_deg))
        )
        turning = valid_agent & (target_turn > threshold)
        turning_f = turning.to(error.dtype)
        result.update(
            turn_agent_count=turning_f.sum(),
            turn_covered_sum=(
                (sample_turn > threshold).any(dim=0) & turning
            ).to(error.dtype).sum(),
            turn_minade_sum=(ade.min(dim=0).values * turning_f).sum(),
            turn_minfde_sum=(fde.min(dim=0).values * turning_f).sum(),
        )

    if full_target.shape[1] >= 4 and steps > 1:
        consecutive = mask[:, 1:] & mask[:, :-1]
        pair_count = consecutive.sum().to(error.dtype)
        pos_scale = position_scale.to(full_target).view(-1, 1, 1, 1)
        vel_scale = (
            velocity_scale.to(full_target).view(-1, 1, 1, 1)
            if velocity_scale is not None
            else torch.full_like(pos_scale, 15.0)
        )
        target_implied_v = (
            (full_target[:, :2, 1:] - full_target[:, :2, :-1])
            * pos_scale
            / float(timestep_seconds)
        )
        target_v = full_target[:, 2:4, 1:] * vel_scale
        target_kinematic = torch.linalg.vector_norm(
            target_implied_v - target_v, dim=1
        )

        sample_pos_scale = position_scale.to(full_samples).view(1, -1, 1, 1, 1)
        sample_vel_scale = (
            velocity_scale.to(full_samples).view(1, -1, 1, 1, 1)
            if velocity_scale is not None
            else torch.full_like(sample_pos_scale, 15.0)
        )
        sample_implied_v = (
            (full_samples[:, :, :2, 1:] - full_samples[:, :, :2, :-1])
            * sample_pos_scale
            / float(timestep_seconds)
        )
        sample_v = full_samples[:, :, 2:4, 1:] * sample_vel_scale
        sample_kinematic = torch.linalg.vector_norm(
            sample_implied_v - sample_v, dim=2
        )
        result.update(
            target_kinematic_sum=(
                target_kinematic * consecutive.to(target_kinematic.dtype)
            ).sum(),
            target_kinematic_count=pair_count,
            sample_kinematic_sum=(
                sample_kinematic
                * consecutive.unsqueeze(0).to(sample_kinematic.dtype)
            ).sum(),
            sample_kinematic_count=pair_count * full_samples.shape[0],
        )

    return result


def evaluate_validation(model_single, dataloader, args, z_shapes, device, is_distributed=False):
    totals = torch.zeros(16, dtype=torch.float64, device=device)
    max_batches = int(args.val_max_batches) if args.val_max_batches > 0 else None
    with preserving_rng_state(), evaluating(model_single):
        for batch_index, batch_raw in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = process_batch(batch_raw, device, train_mode=args.train_mode)
            condition = (
                batch["labels"]
                if uses_label_condition(args.train_mode, args.label_source)
                else None
            )
            log_p, logdet, _ = model_single(
                batch["target_data"],
                condition=condition,
                **build_model_context_kwargs(
                    batch, train_mode=args.train_mode, model=model_single
                ),
            )
            nll = -(log_p + logdet)
            dims = valid_dimension_count(batch, args.in_channel)
            totals[0] += nll.double().sum()
            totals[1] += dims.double().sum()
            totals[2] += nll.numel()

            if args.train_mode == "prediction" and args.val_num_modes > 0:
                reverse_kwargs = build_model_context_kwargs(
                    batch,
                    train_mode=args.train_mode,
                    model=model_single,
                    method_name="reverse",
                    for_sampling=True,
                    future_steps=args.future_steps,
                )
                generation_timestep_mask = build_generation_timestep_mask(
                    batch,
                    train_mode=args.train_mode,
                    future_steps=args.future_steps,
                )
                samples = []
                for mode_sample_index in range(args.val_num_modes):
                    z = sample_latents(
                        batch["target_data"].shape[0],
                        z_shapes,
                        device,
                        args.temp,
                        args.temp_block_decay,
                    )
                    route_reverse_kwargs = {}
                    if getattr(args, "model_arch", "mapglow") == "route_mixture_glow":
                        route_reverse_kwargs["mode_index"] = mode_sample_index % args.route_modes
                    sample = model_single.reverse(z, **route_reverse_kwargs, **reverse_kwargs)
                    samples.append(
                        decode_prediction_states(
                            sample,
                            batch,
                            args,
                            timestep_mask=generation_timestep_mask,
                        )
                    )
                samples = torch.stack(samples, dim=0)
                absolute_target = decode_prediction_states(
                    batch["target_data"], batch, args
                )
                scale = batch["scene_stats"][:, 2].abs()
                scale = torch.where(
                    scale > 0,
                    scale,
                    torch.full_like(scale, float(args.vis_position_scale)),
                )
                metric = trajectory_metric_sums(
                    samples,
                    absolute_target,
                    batch["loss_timestep_mask"],
                    batch["context_agent_mask"],
                    scale,
                    history=batch.get("history_data"),
                    history_timestep_mask=batch.get("history_timestep_mask"),
                    velocity_scale=batch.get("state_scales")[:, 1],
                    yaw_scale=batch.get("state_scales")[:, 2],
                    turn_angle_threshold_deg=args.turn_angle_threshold_deg,
                )
                totals[3] += metric["ade_sum"].double()
                totals[4] += metric["fde_sum"].double()
                totals[5] += metric["minade_sum"].double()
                totals[6] += metric["minfde_sum"].double()
                totals[7] += metric["agent_count"].double()
                totals[8] += metric.get("turn_covered_sum", 0.0)
                totals[9] += metric.get("turn_agent_count", 0.0)
                totals[10] += metric.get("turn_minade_sum", 0.0)
                totals[11] += metric.get("turn_minfde_sum", 0.0)
                totals[12] += metric.get("sample_kinematic_sum", 0.0)
                totals[13] += metric.get("sample_kinematic_count", 0.0)
                totals[14] += metric.get("target_kinematic_sum", 0.0)
                totals[15] += metric.get("target_kinematic_count", 0.0)

    if is_distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if totals[2] <= 0 or totals[1] <= 0:
        raise RuntimeError("validation loader produced no valid scenes/dimensions")
    metrics = {
        "nll": float((totals[0] / totals[2]).item()),
        "nll_per_valid_dim": float((totals[0] / totals[1]).item()),
    }
    if totals[7] > 0:
        metrics.update(
            ade=float((totals[3] / totals[7]).item()),
            fde=float((totals[4] / totals[7]).item()),
            minade=float((totals[5] / totals[7]).item()),
            minfde=float((totals[6] / totals[7]).item()),
        )
    if totals[9] > 0:
        metrics.update(
            turn_recall=float((totals[8] / totals[9]).item()),
            turn_minade=float((totals[10] / totals[9]).item()),
            turn_minfde=float((totals[11] / totals[9]).item()),
            turn_agent_count=int(totals[9].item()),
        )
    if totals[13] > 0 and totals[15] > 0:
        metrics.update(
            sample_kinematic_error=float((totals[12] / totals[13]).item()),
            target_kinematic_error=float((totals[14] / totals[15]).item()),
        )
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError(f"non-finite validation metrics: {metrics}")
    return metrics


def validate_training_args(args):
    positive_ints = ("batch", "epochs", "n_flow", "n_block", "n_modes")
    for name in positive_ints:
        value = int(getattr(args, name))
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
    if args.route_modes <= 0 or args.route_modes > 9:
        raise ValueError("route_modes must be in [1,9] for the route-id conditioner embedding")
    if args.route_hidden_dim <= 0 or args.route_hidden_dim % 8 != 0:
        raise ValueError("route_hidden_dim must be positive and divisible by 8")
    if args.mixture_chunk_size <= 0:
        raise ValueError("mixture_chunk_size must be positive")
    for name in (
        "proposal_loss_weight", "route_classification_loss_weight",
        "proposal_smoothness_loss_weight", "proposal_diversity_loss_weight",
        "proposal_map_compliance_loss_weight",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if args.model_arch == "route_mixture_glow":
        if args.train_mode != "prediction" or args.prediction_representation != "cv_residual":
            raise ValueError("route_mixture_glow requires prediction mode with cv_residual targets")
        if args.n_modes < args.route_modes:
            raise ValueError("n_modes must cover every route component")
    if args.start_epoch < 0 or args.start_iter < 0:
        raise ValueError("start_epoch and start_iter must be >= 0")
    if args.lr <= 0 or not np.isfinite(args.lr):
        raise ValueError(f"lr must be finite and > 0, got {args.lr}")
    if getattr(args, "warmup_steps", 0) < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {args.warmup_steps}")
    min_lr_ratio = float(getattr(args, "min_lr_ratio", 0.05))
    if not np.isfinite(min_lr_ratio) or not 0.0 < min_lr_ratio <= 1.0:
        raise ValueError(
            f"min_lr_ratio must be finite and in (0, 1], got {min_lr_ratio}"
        )
    if args.grad_clip_norm <= 0 or not np.isfinite(args.grad_clip_norm):
        raise ValueError(f"grad_clip_norm must be finite and > 0, got {args.grad_clip_norm}")
    if args.amp_init_scale <= 0 or not np.isfinite(args.amp_init_scale):
        raise ValueError(
            f"amp_init_scale must be finite and > 0, got {args.amp_init_scale}"
        )
    if not 0.0 <= args.label_keep_prob <= 1.0:
        raise ValueError(f"label_keep_prob must be in [0, 1], got {args.label_keep_prob}")
    if args.temp < 0 or args.temp_block_decay <= 0:
        raise ValueError("temp must be >= 0 and temp_block_decay must be > 0")
    if not np.isfinite(args.cfg_scale) or args.cfg_scale < 0:
        raise ValueError(f"cfg_scale must be finite and >= 0, got {args.cfg_scale}")
    if args.turn_angle_threshold_deg <= 0 or args.stationary_dist_threshold < 0:
        raise ValueError(
            "turn_angle_threshold_deg must be > 0 and stationary_dist_threshold must be >= 0"
        )
    if args.img_size_h == 0 or args.img_size_h < -1 or args.img_size_w == 0 or args.img_size_w < -1:
        raise ValueError("img_size_h/img_size_w must be -1 (infer) or positive")
    if args.devices == 0 or args.devices < -1:
        raise ValueError("devices must be -1 (all visible) or positive")
    if args.vis_position_scale <= 0 or not np.isfinite(args.vis_position_scale):
        raise ValueError("vis_position_scale must be finite and > 0")
    if args.history_steps <= 0 or args.future_steps <= 0:
        raise ValueError("history_steps and future_steps must be > 0")
    if args.prediction_target_steps < args.future_steps:
        raise ValueError("prediction_target_steps must be >= future_steps")
    if args.train_mode == "prediction" and args.in_channel != 5:
        raise ValueError(
            "prediction mode models only the five dynamic channels (x,y,vx,vy,yaw); "
            "length/width are supplied as static_dimensions context"
        )
    if getattr(args, "prediction_representation", "absolute") not in (
        "absolute", "delta", "cv_residual",
    ):
        raise ValueError(
            "prediction_representation must be absolute, delta, or cv_residual"
        )
    if args.train_mode == "prediction" and args.prediction_target_steps % (2 ** args.n_block) != 0:
        raise ValueError(
            "prediction_target_steps must be divisible by 2**n_block. "
            f"Got prediction_target_steps={args.prediction_target_steps}, n_block={args.n_block}."
        )
    for name in (
        "sample_interval", "save_interval", "save_epoch_interval", "val_interval",
        "val_max_batches", "num_workers", "prefetch_factor",
    ):
        minimum = -1 if name == "val_max_batches" else 0
        if int(getattr(args, name)) < minimum:
            raise ValueError(f"{name} has invalid negative value {getattr(args, name)}")
    if args.val_batch < 0 or args.val_num_modes < 0:
        raise ValueError("val_batch and val_num_modes must be >= 0")
    if args.sample_interval > 0 and not args.val_combined_path:
        raise ValueError(
            "sample_interval > 0 requires val_combined_path because periodic "
            "inference is not allowed to use training batches"
        )


def train(local_rank, world_size, args, rank=None):
    rank = local_rank if rank is None else rank
    is_distributed = world_size > 1
    set_random_seed(args.seed, rank=rank)

    if torch.cuda.is_available():
        if is_distributed:
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if is_distributed:
        dist.init_process_group(
            backend=args.dist_backend or ("nccl" if torch.cuda.is_available() else "gloo"),
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

    is_main = (not is_distributed) or rank == 0

    dataloader, sampler, dataset = build_dataloader(args, is_distributed, rank, world_size)
    args.label_source = dataset.label_source
    val_dataloader = None
    val_sampler = None
    fixed_sample_batch_raw = None
    random_val_sample_dataloader = None
    random_val_sample_iterator = None
    if args.val_combined_path:
        val_dataloader, val_sampler, val_dataset = build_dataloader(
            args,
            is_distributed,
            rank,
            world_size,
            combined_path=args.val_combined_path,
            shuffle=False,
            batch_size=args.val_batch if args.val_batch > 0 else args.batch,
        )
        if (
            val_dataset.input_channels != dataset.input_channels
            or val_dataset.time_steps != dataset.time_steps
            or val_dataset.max_agents != dataset.max_agents
        ):
            raise ValueError("validation dataset target shape must match the training dataset")
        if is_main:
            if getattr(args, "fixed_validation_samples", True):
                fixed_sample_batch_raw = next(iter(val_dataloader))
            else:
                random_val_sample_dataloader = (
                    build_random_validation_sample_dataloader(val_dataset, args)
                )

    if args.img_size_h <= 0:
        args.img_size_h = dataset.time_steps
    elif args.img_size_h != dataset.time_steps:
        raise ValueError(
            f"img_size_h={args.img_size_h} does not match dataset target time steps={dataset.time_steps}. "
            "Use -1 to infer it from the selected train_mode."
        )
    if args.img_size_w <= 0:
        args.img_size_w = dataset.max_agents
    elif args.img_size_w != dataset.max_agents:
        raise ValueError(
            f"img_size_w={args.img_size_w} does not match dataset max agents={dataset.max_agents}. "
            "Use -1 to infer it from the dataset."
        )

    model_arch = getattr(args, "model_arch", "mapglow")
    model_class = (
        RouteMixtureGlow
        if model_arch == "route_mixture_glow"
        else Glow
    )
    glow_kwargs = dict(
        in_channel=args.in_channel,
        condition_dim=32,
        n_flow=args.n_flow,
        n_block=args.n_block,
        affine=args.affine,
        conv_lu=not args.no_lu,
    )
    if "history_input_dim" in inspect.signature(model_class.__init__).parameters:
        glow_kwargs["history_input_dim"] = args.in_channel
    if model_arch == "route_mixture_glow":
        glow_kwargs.update(
            route_modes=args.route_modes,
            future_steps=args.prediction_target_steps,
            hidden_dim=args.route_hidden_dim,
            mixture_chunk_size=args.mixture_chunk_size,
        )
    model_single = model_class(**glow_kwargs)
    model_single.to(device)

    optimizer = torch.optim.AdamW(model_single.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = build_lr_scheduler(optimizer, args, len(dataloader))

    use_amp = args.amp and torch.cuda.is_available()
    scaler = make_grad_scaler(
        enabled=use_amp,
        init_scale=getattr(args, "amp_init_scale", 256.0),
    )

    resume_checkpoint = None
    if args.resume_path:
        resume_checkpoint = load_training_checkpoint(
            args.resume_path,
            model_single,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            load_optimizer=not args.resume_model_only,
            allow_partial=args.allow_partial_checkpoint,
        )
    if args.loadckpt:
        if args.load_model_path:
            ckpt = safe_load_state(args.load_model_path, map_location="cpu")
            if ckpt is not None:
                state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
                load_model_state_checked(
                    model_single,
                    state,
                    args.load_model_path,
                    allow_partial=args.allow_partial_checkpoint,
                )
        if args.load_optim_path:
            opt_state = safe_load_state(args.load_optim_path, map_location="cpu")
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)

    resume_epoch_step = 0
    if args.resume_path and resume_checkpoint is not None and not args.no_resume_iter:
        args.start_iter = int(
            resume_checkpoint.get(
                "next_iter",
                resume_checkpoint.get("global_step", resume_checkpoint.get("step", -1) + 1),
            )
        )
        args.start_epoch = int(resume_checkpoint.get("next_epoch", resume_checkpoint.get("epoch", args.start_epoch)))
        resume_epoch_step = int(resume_checkpoint.get("next_epoch_step", 0))
        if is_main:
            print(
                "[checkpoint] resume position set to "
                f"epoch={args.start_epoch}, epoch_step={resume_epoch_step}, global_step={args.start_iter}"
            )

    if args.compile and hasattr(torch, "compile"):
        if is_main:
            print(f"[rank {rank}] compiling model with torch.compile")
        model_single = torch.compile(model_single, mode="reduce-overhead")

    initialize_model_once(
        model_single,
        dataloader,
        device,
        is_distributed=is_distributed,
        local_rank=rank,
        sampler=sampler,
        train_mode=args.train_mode,
    )
    if resume_checkpoint is not None and not args.resume_model_only:
        restored = restore_rng_state(resume_checkpoint)
        if is_main:
            print(f"[checkpoint] restored Python/NumPy/Torch RNG state={restored}")

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model_single,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=args.ddp_find_unused_parameters,
        )
    else:
        model = model_single

    z_shapes = calc_z_shapes(args.in_channel, args.img_size_h, args.img_size_w, args.n_block)

    writer = None
    metrics_path = None
    summary_path = None
    if is_main:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        logdir = Path(args.log_dir) / f"run_interaction_combined_{run_id}"
        logdir.mkdir(parents=True, exist_ok=True)
        if SummaryWriter is not None:
            writer = SummaryWriter(log_dir=str(logdir))
            print(f"[rank {rank}] TensorBoard logdir: {logdir}")
        else:
            print("[tensorboard] tensorboard is not installed; scalar logging disabled")
        metrics_path = (
            Path(args.metrics_out_path)
            if args.metrics_out_path
            else Path(args.ckpt_dir) / "training_metrics.jsonl"
        )
        summary_path = Path(args.ckpt_dir) / "training_summary.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if not args.resume_path:
            metrics_path.write_text("", encoding="utf-8")
        print(f"[metrics] JSONL: {metrics_path}")

    steps_per_epoch = len(dataloader)
    global_step = int(args.start_iter)
    max_steps = int(args.max_steps) if args.max_steps is not None and args.max_steps > 0 else None
    if is_main:
        limit_msg = f", max_steps={max_steps}" if max_steps is not None else ""
        print(
            f"[train] epoch-based loop: epochs={args.epochs}, "
            f"start_epoch={args.start_epoch}, steps_per_epoch={steps_per_epoch}, "
            f"start_global_step={global_step}{limit_msg}"
        )

    stop_training = False
    best_val_nll = float(
        resume_checkpoint.get("best_val_nll", float("inf"))
        if resume_checkpoint is not None else float("inf")
    )
    latest_validation_metrics = (
        resume_checkpoint.get("validation_metrics")
        if resume_checkpoint is not None else None
    )
    last_checkpoint_global_step = -1
    final_epoch = int(args.start_epoch)
    final_epoch_step = -1
    epochs_started = 0
    epochs_fully_completed = 0
    if max_steps is not None and global_step >= max_steps:
        stop_training = True
        if is_main:
            print(
                f"[train] start_global_step={global_step} already reached max_steps={max_steps}; "
                "no optimizer update is run"
            )
    for epoch in range(int(args.start_epoch), int(args.epochs)):
        if stop_training:
            break
        epochs_started += 1
        final_epoch = epoch
        if sampler is not None:
            sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        iterable = enumerate(dataloader)
        show_progress = is_main and not getattr(args, "disable_tqdm", False)
        if show_progress:
            progress = tqdm(
                iterable,
                total=steps_per_epoch,
                ncols=120,
                desc=f"Epoch {epoch + 1}/{args.epochs}",
            )
        else:
            progress = iterable

        last_epoch_step = None
        for epoch_step, batch_raw in progress:
            if epoch == int(args.start_epoch) and epoch_step < resume_epoch_step:
                continue

            i = global_step

            batch = process_batch(batch_raw, device, train_mode=args.train_mode)
            target_data = batch["target_data"]
            labels = batch["labels"]
            map_data = batch["map_data"]

            bad_input = (not torch.isfinite(map_data).all()) or (not torch.isfinite(target_data).all())
            if distributed_bad_flag(bad_input, is_distributed, device):
                if is_main:
                    print(f"[Step {i}] NaN detected in input tensors, skip batch on all ranks")
                continue

            use_conditional = distributed_label_condition(
                args.train_mode,
                args.label_source,
                args.label_keep_prob,
                is_distributed,
                device,
            )
            condition_input = labels if use_conditional else None

            with cuda_autocast(enabled=use_amp):
                log_p, logdet, model_aux = model(
                    target_data,
                    condition=condition_input,
                    **build_model_context_kwargs(
                        batch, train_mode=args.train_mode, model=model
                    ),
                )
                nll_per_scene = -(logdet + log_p)
                valid_dims = valid_dimension_count(batch, args.in_channel)
                nll_per_valid_dim = nll_per_scene / valid_dims
                if args.loss_normalize == "valid_dim":
                    loss_value = nll_per_valid_dim.mean()
                else:
                    loss_value = nll_per_scene.mean()
                if isinstance(model_aux, dict) and "proposal_loss" in model_aux:
                    loss_value = (
                        loss_value
                        + args.proposal_loss_weight * model_aux["proposal_loss"]
                        + args.route_classification_loss_weight * model_aux["route_classification_loss"]
                        + args.proposal_smoothness_loss_weight * model_aux["proposal_smoothness_loss"]
                        + args.proposal_diversity_loss_weight * model_aux["proposal_diversity_loss"]
                        + args.proposal_map_compliance_loss_weight * model_aux["proposal_map_compliance_loss"]
                    )

            bad_loss = (
                not torch.isfinite(loss_value)
                or not torch.isfinite(log_p).all()
                or not torch.isfinite(logdet).all()
            )
            if distributed_bad_flag(bad_loss, is_distributed, device):
                if is_main:
                    print(f"[Step {i}] NaN/Inf detected, skip batch on all ranks")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.grad_clip_norm
            )
            bad_grad = (not torch.isfinite(grad_norm).item()) or not gradients_are_finite(
                model.parameters()
            )
            if distributed_bad_flag(bad_grad, is_distributed, device):
                if is_main:
                    print(f"[Step {i}] NaN/Inf gradient detected, skip optimizer step on all ranks")
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.update(new_scale=max(float(scaler.get_scale()) * 0.5, 1.0))
                continue

            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            new_scale = scaler.get_scale()
            if old_scale <= new_scale:
                scheduler.step()

            completed_step = global_step + 1
            if is_main:
                log_p_mean = log_p.mean()
                logdet_mean = logdet.mean()
                nll_dim_mean = nll_per_valid_dim.mean()
                if show_progress:
                    progress.set_description(
                        f"Epoch {epoch + 1}/{args.epochs}; step: {completed_step}; "
                        f"Loss: {loss_value.item():.5f}; logP: {log_p_mean.item():.5f}; "
                        f"logdet: {logdet_mean.item():.5f}; "
                        f"nll/dim: {nll_dim_mean.item():.5f}; lr: {optimizer.param_groups[0]['lr']:.7f}"
                    )
                if writer is not None:
                    writer.add_scalar("train/loss", loss_value.item(), global_step)
                    writer.add_scalar("train/log_p", log_p_mean.item(), global_step)
                    writer.add_scalar("train/logdet", logdet_mean.item(), global_step)
                    writer.add_scalar("train/nll_per_valid_dim", nll_dim_mean.item(), global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("train/epoch", epoch, global_step)
                    if isinstance(model_aux, dict) and "proposal_loss" in model_aux:
                        writer.add_scalar("train/proposal_loss", model_aux["proposal_loss"].item(), global_step)
                        writer.add_scalar("train/route_classification_loss", model_aux["route_classification_loss"].item(), global_step)
                        writer.add_scalar("train/proposal_smoothness_loss", model_aux["proposal_smoothness_loss"].item(), global_step)
                        writer.add_scalar("train/proposal_diversity_loss", model_aux["proposal_diversity_loss"].item(), global_step)
                        writer.add_scalar("train/proposal_map_compliance_loss", model_aux["proposal_map_compliance_loss"].item(), global_step)
                        route_entropy = -(
                            model_aux["route_probabilities"]
                            * model_aux["route_probabilities"].clamp_min(1e-9).log()
                        ).sum(-1).mean()
                        writer.add_scalar("train/route_entropy", route_entropy.item(), global_step)
                route_metrics = {}
                if isinstance(model_aux, dict) and "proposal_loss" in model_aux:
                    route_prob = model_aux["route_probabilities"]
                    route_entropy = -(route_prob * route_prob.clamp_min(1e-9).log()).sum(-1).mean()
                    route_metrics = {
                        "proposal_loss": float(model_aux["proposal_loss"].item()),
                        "route_classification_loss": float(model_aux["route_classification_loss"].item()),
                        "proposal_smoothness_loss": float(model_aux["proposal_smoothness_loss"].item()),
                        "proposal_diversity_loss": float(model_aux["proposal_diversity_loss"].item()),
                        "proposal_map_compliance_loss": float(model_aux["proposal_map_compliance_loss"].item()),
                        "route_entropy": float(route_entropy.item()),
                        "effective_route_modes": float(torch.exp(route_entropy).item()),
                    }
                append_jsonl(
                    metrics_path,
                    {
                        "type": "train_step",
                        "epoch": int(epoch),
                        "epoch_step": int(epoch_step),
                        "global_step": int(completed_step),
                        "loss": float(loss_value.item()),
                        "log_p": float(log_p_mean.item()),
                        "logdet": float(logdet_mean.item()),
                        "nll_per_valid_dim": float(nll_dim_mean.item()),
                        "grad_norm": float(grad_norm.item()),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        **route_metrics,
                    },
                )

            if is_main and args.sample_interval > 0 and completed_step % args.sample_interval == 0:
                with torch.no_grad():
                    sample_batch_raw, random_val_sample_iterator = (
                        next_validation_sample_batch(
                            fixed_sample_batch_raw,
                            random_val_sample_dataloader,
                            random_val_sample_iterator,
                        )
                    )
                    sample_batch = process_batch(
                        sample_batch_raw,
                        device,
                        train_mode=args.train_mode,
                    )
                    save_samples_npz(
                        model_single,
                        sample_batch,
                        args,
                        z_shapes,
                        device,
                        completed_step - 1,
                    )

            if is_main and args.save_interval > 0 and completed_step % args.save_interval == 0:
                ckpt_dir = Path(args.ckpt_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                last_path = ckpt_dir / "last.pt"
                save_training_checkpoint(
                    last_path,
                    model_single,
                    optimizer,
                    scheduler,
                    scaler,
                    args,
                    completed_step - 1,
                    epoch,
                    epoch_step,
                    steps_per_epoch,
                    extra_state={
                        "best_val_nll": best_val_nll,
                        "validation_metrics": latest_validation_metrics,
                    },
                )
                if args.save_step_checkpoints:
                    step_path = ckpt_dir / f"step_{completed_step:06d}.pt"
                    save_training_checkpoint(
                        step_path,
                        model_single,
                        optimizer,
                        scheduler,
                        scaler,
                        args,
                        completed_step - 1,
                        epoch,
                        epoch_step,
                        steps_per_epoch,
                        extra_state={
                            "best_val_nll": best_val_nll,
                            "validation_metrics": latest_validation_metrics,
                        },
                    )
                last_checkpoint_global_step = completed_step
                print(f"[rank {rank}] saved checkpoint: {last_path}")

            if is_main and args.keep_legacy_checkpoints and args.save_interval > 0 and completed_step % args.save_interval == 0:
                model_path = ckpt_dir / "model_interaction_combined.pt"
                optim_path = ckpt_dir / "optim_interaction_combined.pt"
                torch.save(unwrap_model(model_single).state_dict(), model_path)
                torch.save(optimizer.state_dict(), optim_path)
                print(f"[rank {rank}] saved legacy checkpoints: {model_path}, {optim_path}")

            global_step = completed_step
            last_epoch_step = epoch_step
            final_epoch_step = epoch_step
            if max_steps is not None and global_step >= max_steps:
                stop_training = True
                break

        resume_epoch_step = 0
        validation_metrics = None
        should_validate = (
            val_dataloader is not None
            and args.val_interval > 0
            and (((epoch + 1) % args.val_interval == 0) or stop_training)
        )
        if should_validate:
            validation_metrics = evaluate_validation(
                model_single,
                val_dataloader,
                args,
                z_shapes,
                device,
                is_distributed=is_distributed,
            )
            latest_validation_metrics = validation_metrics
            if is_main:
                # A same-step checkpoint written before validation has stale
                # best/metric metadata and must be refreshed below.
                last_checkpoint_global_step = -1
                print(
                    "[validation] "
                    + ", ".join(f"{key}={value:.6f}" for key, value in validation_metrics.items())
                )
                if writer is not None:
                    for key, value in validation_metrics.items():
                        writer.add_scalar(f"val/{key}", value, global_step)
                append_jsonl(
                    metrics_path,
                    {
                        "type": "validation_epoch",
                        "epoch": int(epoch),
                        "global_step": int(global_step),
                        **{key: float(value) for key, value in validation_metrics.items()},
                    },
                )
                if validation_metrics["nll_per_valid_dim"] < best_val_nll:
                    best_val_nll = validation_metrics["nll_per_valid_dim"]
                    best_path = Path(args.ckpt_dir) / "best.pt"
                    save_training_checkpoint(
                        best_path,
                        model_single,
                        optimizer,
                        scheduler,
                        scaler,
                        args,
                        global_step - 1,
                        epoch,
                        last_epoch_step if last_epoch_step is not None else -1,
                        steps_per_epoch,
                        extra_state={
                            "best_val_nll": best_val_nll,
                            "validation_metrics": validation_metrics,
                        },
                    )
                    print(f"[validation] saved best checkpoint: {best_path}")

        if (
            is_main
            and not stop_training
            and last_epoch_step is not None
            and args.save_epoch_interval > 0
            and (epoch + 1) % args.save_epoch_interval == 0
        ):
            ckpt_dir = Path(args.ckpt_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            last_path = ckpt_dir / "last.pt"
            save_training_checkpoint(
                last_path,
                model_single,
                optimizer,
                scheduler,
                scaler,
                args,
                global_step - 1,
                epoch,
                last_epoch_step,
                steps_per_epoch,
                extra_state={
                    "best_val_nll": best_val_nll,
                    "validation_metrics": validation_metrics,
                },
            )
            last_checkpoint_global_step = global_step
            print(f"[rank {rank}] saved epoch checkpoint: {last_path}")
        if stop_training:
            break
        epochs_fully_completed += 1

    # max_steps may stop between periodic/epoch checkpoints. Always persist the
    # exact final optimizer and RNG state after at least one completed update.
    if is_main and global_step > int(args.start_iter) and last_checkpoint_global_step != global_step:
        last_path = Path(args.ckpt_dir) / "last.pt"
        save_training_checkpoint(
            last_path,
            model_single,
            optimizer,
            scheduler,
            scaler,
            args,
            global_step - 1,
            final_epoch,
            final_epoch_step,
            steps_per_epoch,
            extra_state={
                "best_val_nll": best_val_nll,
                "validation_metrics": latest_validation_metrics,
            },
        )
        print(f"[rank {rank}] saved final checkpoint: {last_path}")

    if is_main:
        write_json_atomic(
            summary_path,
            {
                "status": "complete",
                "train_mode": args.train_mode,
                "epochs_started": int(epochs_started),
                "epochs_fully_completed": int(epochs_fully_completed),
                "global_step": int(global_step),
                "best_val_nll_per_valid_dim": (
                    None if not np.isfinite(best_val_nll) else float(best_val_nll)
                ),
                "metrics_jsonl": str(metrics_path),
                "last_checkpoint": str(Path(args.ckpt_dir) / "last.pt"),
                "dropped_future_only_agents": int(dataset.dropped_future_only_agents),
                "dropped_future_only_timesteps": int(dataset.dropped_future_only_points),
            },
        )
        print(f"[metrics] summary: {summary_path}")

    if writer is not None:
        writer.close()
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train MapGlow on combined INTERACTION NPZ")
    parser.add_argument("--config", default="", type=str, help="optional YAML config path")
    parser.add_argument(
        "--combined_path",
        default="",
        type=str,
        help="path to combined interaction npz",
    )
    parser.add_argument("--batch", default=8, type=int, help="per-GPU batch size")
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
    parser.add_argument("--start_epoch", default=0, type=int, help="epoch index to start from")
    parser.add_argument("--max_steps", default=-1, type=int, help="optional global-step cap; <=0 trains full epochs")
    parser.add_argument("--iter", default=-1, type=int, help="deprecated alias for --max_steps")
    parser.add_argument("--start_iter", default=0, type=int, help="global step to start logging/checkpoint numbering")
    parser.add_argument("--n_flow", default=16, type=int, help="number of flows per block")
    parser.add_argument("--n_block", default=3, type=int, help="number of blocks")
    parser.add_argument("--model_arch", default="mapglow", choices=["mapglow", "route_mixture_glow"],
                        help="model architecture; route_mixture_glow requires v5 LaneGraph data")
    parser.add_argument("--route_modes", default=6, type=int, help="route mixture component count")
    parser.add_argument("--route_hidden_dim", default=256, type=int, help="query-centric encoder width")
    parser.add_argument("--mixture_chunk_size", default=2, type=int,
                        help="number of route components evaluated together")
    parser.add_argument("--proposal_loss_weight", default=0.2, type=float)
    parser.add_argument("--route_classification_loss_weight", default=0.05, type=float)
    parser.add_argument("--proposal_smoothness_loss_weight", default=0.01, type=float)
    parser.add_argument("--proposal_diversity_loss_weight", default=0.1, type=float)
    parser.add_argument("--proposal_map_compliance_loss_weight", default=0.05, type=float)
    parser.add_argument("--no_lu", action="store_true", help="disable LU conv")
    parser.add_argument("--affine", action="store_true", help="use affine coupling")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--scheduler_type", default="step", choices=["step", "cosine"],
                        help="learning-rate schedule")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="linear warmup steps used by the cosine scheduler")
    parser.add_argument("--min_lr_ratio", default=0.05, type=float,
                        help="final/base LR ratio used by the cosine scheduler")
    parser.add_argument("--img_size_h", default=-1, type=int, help="time dimension, -1 means infer from dataset")
    parser.add_argument("--img_size_w", default=-1, type=int, help="agent dimension, -1 means infer from dataset")
    parser.add_argument(
        "--in_channel", default=None, type=int, choices=[5, 7],
        help="target channels; defaults to 5 for prediction and 7 for initialization",
    )
    parser.add_argument("--train_mode", default="initialization", choices=["initialization", "prediction"],
                        help="initialization trains on full 40-frame target; prediction trains on future target conditioned on history")
    parser.add_argument("--use_history", action=argparse.BooleanOptionalAction, default=False,
                        help="deprecated compatibility flag; --use_history maps to --train_mode prediction")
    parser.add_argument("--history_steps", default=10, type=int, help="history steps used in prediction mode")
    parser.add_argument("--future_steps", default=30, type=int, help="real future steps used in prediction mode")
    parser.add_argument("--prediction_target_steps", default=32, type=int,
                        help="padded target length for prediction mode; must be divisible by 2**n_block")
    parser.add_argument(
        "--prediction_representation",
        default="absolute",
        choices=["absolute", "delta", "cv_residual"],
        help=(
            "prediction target representation; cv_residual models a heading-aligned "
            "residual around a constant-motion history baseline"
        ),
    )
    parser.add_argument("--label_source", default="auto", choices=["auto", "none", "dataset", "target"],
                        help="label semantics: prediction always uses none; auto=dataset for initialization")
    parser.add_argument(
        "--allow_legacy_prediction_data",
        action="store_true",
        help="allow prediction NPZ without forecasting_safe=True (known future-leakage risk)",
    )
    parser.add_argument("--turn_angle_threshold_deg", default=30.0, type=float,
                        help="heading-change threshold used when inferring target labels")
    parser.add_argument("--stationary_dist_threshold", default=0.0, type=float,
                        help="normalized displacement threshold for stationary target labels; 0 disables stationary override")
    parser.add_argument("--label_keep_prob", default=0.8, type=float, help="probability of keeping label condition")
    parser.add_argument("--temp", default=0.7, type=float, help="sampling temperature")
    parser.add_argument("--temp_block_decay", default=1.0, type=float, help="per-block latent temp decay")
    parser.add_argument("--cfg_scale", default=1.0, type=float, help="CFG scale for label-conditioned sampling")
    parser.add_argument("--n_modes", default=6, type=int, help="number of samples to save per batch")
    parser.add_argument("--num_workers", default=4, type=int, help="dataloader workers")
    parser.add_argument("--worker_start_method", default="", choices=["", "fork", "spawn", "forkserver"],
                        help="dataloader multiprocessing context; empty uses PyTorch default")
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True,
                        help="keep dataloader workers alive when num_workers > 0")
    parser.add_argument("--prefetch_factor", default=2, type=int,
                        help="dataloader prefetch factor when num_workers > 0; <=0 disables explicit setting")
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="disable interactive progress output; metrics are still written to JSONL/TensorBoard",
    )
    parser.add_argument("--sample_interval", default=2000, type=int, help="sample save interval")
    parser.add_argument("--sample_seed", default=12345, type=int,
                        help="fixed latent seed for comparable periodic samples; negative disables")
    parser.add_argument("--fixed_validation_samples", action=argparse.BooleanOptionalAction, default=True,
                        help=(
                            "use the same first validation batch for periodic samples; "
                            "when false, draw shuffled batches from validation only"
                        ))
    parser.add_argument("--save_interval", default=2000, type=int, help="checkpoint save interval")
    parser.add_argument("--save_epoch_interval", default=1, type=int, help="save last.pt every N epochs; <=0 disables")
    parser.add_argument("--val_combined_path", default="", type=str,
                        help="optional independent validation NPZ")
    parser.add_argument("--val_batch", default=0, type=int,
                        help="validation batch size; 0 reuses --batch")
    parser.add_argument("--val_interval", default=1, type=int,
                        help="validate every N epochs; 0 disables")
    parser.add_argument("--val_num_modes", default=6, type=int,
                        help="prediction samples per validation scene for ADE/FDE")
    parser.add_argument("--val_max_batches", default=-1, type=int,
                        help="optional validation batch cap; -1 evaluates all")
    parser.add_argument("--log_dir", default="./runs", type=str, help="tensorboard log dir")
    parser.add_argument("--ckpt_dir", default="./results", type=str, help="checkpoint dir")
    parser.add_argument("--sample_out_dir", default="./results", type=str, help="sample output dir")
    parser.add_argument("--metrics_out_path", default="", type=str,
                        help="JSONL metric path; default is <ckpt_dir>/training_metrics.jsonl")
    parser.add_argument("--save_sample_images", action=argparse.BooleanOptionalAction, default=True,
                        help="save visualization pngs together with sample npz")
    parser.add_argument("--vis_num_scenes", default=4, type=int, help="number of sampled scenes to visualize")
    parser.add_argument("--vis_num_modes", default=2, type=int, help="number of modes per scene to visualize")
    parser.add_argument("--vis_position_scale", default=50.0, type=float,
                        help="position scale used to denormalize x/y for visualization")
    parser.add_argument("--vis_pad", default=0.08, type=float, help="fractional padding in normalized visualization limits")
    parser.add_argument("--vis_dpi", default=180, type=int, help="dpi for saved visualization images")
    parser.add_argument("--load_model_path", default="", type=str, help="optional model checkpoint path")
    parser.add_argument("--load_optim_path", default="", type=str, help="optional optimizer checkpoint path")
    parser.add_argument("--loadckpt", action="store_true", help="load ckpt before training")
    parser.add_argument("--resume_path", default="", type=str,
                        help="full training checkpoint path, e.g. results/last.pt")
    parser.add_argument("--resume_model_only", action="store_true",
                        help="with --resume_path, load model weights but not optimizer/scheduler/scaler")
    parser.add_argument("--no_resume_iter", action="store_true",
                        help="with --resume_path, keep CLI/config start_iter instead of checkpoint next_iter")
    parser.add_argument("--save_step_checkpoints", action=argparse.BooleanOptionalAction, default=False,
                        help="also save numbered step_*.pt checkpoints")
    parser.add_argument("--keep_legacy_checkpoints", action=argparse.BooleanOptionalAction, default=True,
                        help="also save model_interaction_combined.pt and optim_interaction_combined.pt")
    parser.add_argument("--allow_partial_checkpoint", action="store_true",
                        help="explicitly allow missing/unexpected checkpoint model keys")
    parser.add_argument("--loss_normalize", default="scene", choices=["scene", "valid_dim"],
                        help="scene keeps original summed scene NLL; valid_dim averages by valid target dimensions")
    parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision")
    parser.add_argument(
        "--amp_init_scale",
        default=256.0,
        type=float,
        help="conservative initial AMP loss scale validated on the full MapGlow model",
    )
    parser.add_argument("--grad_clip_norm", default=5.0, type=float,
                        help="maximum finite gradient norm")
    parser.add_argument("--compile", action="store_true", help="enable torch.compile when available")
    parser.add_argument("--seed", default=42, type=int, help="base random seed; negative disables seeding")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False,
                        help="enable deterministic algorithms where PyTorch supports them")
    parser.add_argument("--devices", default=-1, type=int,
                        help="number of local CUDA devices for spawn launcher; -1 uses all visible GPUs")
    parser.add_argument("--launcher", default="auto", choices=["auto", "spawn", "torchrun", "none"],
                        help="distributed launcher mode")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str, help="MASTER_ADDR for spawn launcher")
    parser.add_argument("--master_port", default="12349", type=str, help="MASTER_PORT for spawn launcher")
    parser.add_argument("--dist_backend", default="", type=str, help="override distributed backend")
    parser.add_argument("--ddp_find_unused_parameters", action=argparse.BooleanOptionalAction, default=True,
                        help="DDP find_unused_parameters; keep true because label conditioning can disable branches")

    config_probe_args, _ = parser.parse_known_args()
    if config_probe_args.config:
        if os.path.exists(config_probe_args.config):
            config_data = load_yaml_config(config_probe_args.config, parser)
            parser.set_defaults(**config_data)
            print(f"[config] loaded: {config_probe_args.config}")
        else:
            print(f"[config] missing config: {config_probe_args.config}, use CLI/defaults")

    args = parser.parse_args()
    if args.in_channel is None:
        args.in_channel = 5 if args.train_mode == "prediction" else 7
    if not os.path.exists(args.combined_path):
        raise FileNotFoundError(f"combined dataset not found: {args.combined_path}")
    if args.val_combined_path and not os.path.exists(args.val_combined_path):
        raise FileNotFoundError(f"validation dataset not found: {args.val_combined_path}")

    if args.iter is not None and args.iter > 0 and args.max_steps <= 0:
        args.max_steps = args.iter
        print(f"[config] --iter is deprecated; using max_steps={args.max_steps}")
    if args.use_history and args.train_mode == "initialization":
        args.train_mode = "prediction"
        print("[config] --use_history is deprecated; switching train_mode to prediction")
    args.use_history = args.train_mode == "prediction"
    if args.num_workers == 0 and not args.persistent_workers:
        args.persistent_workers = False
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    validate_training_args(args)

    print("Training args:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    torchrun_world_size = get_env_int("WORLD_SIZE", 1)
    torchrun_rank = get_env_int("RANK", 0)
    torchrun_local_rank = get_env_int("LOCAL_RANK", 0)
    use_torchrun = args.launcher == "torchrun" or (
        args.launcher == "auto" and torchrun_world_size is not None and torchrun_world_size > 1
    )

    visible_gpus = torch.cuda.device_count()
    if use_torchrun:
        print(
            f"Detected torchrun environment: rank={torchrun_rank}, "
            f"local_rank={torchrun_local_rank}, world_size={torchrun_world_size}"
        )
        train(torchrun_local_rank, torchrun_world_size, args, rank=torchrun_rank)
        return

    if args.launcher == "none":
        print("Launcher disabled, running single process.")
        train(0, 1, args, rank=0)
        return

    os.environ.setdefault("MASTER_ADDR", args.master_addr)
    os.environ.setdefault("MASTER_PORT", args.master_port)

    requested_gpus = visible_gpus if args.devices is None or args.devices < 0 else args.devices
    if visible_gpus == 0 or requested_gpus == 0:
        print("No GPU detected, run in single-process CPU mode.")
        train(0, 1, args, rank=0)
    elif requested_gpus == 1:
        print("Detected 1 GPU, run in single-process CUDA mode.")
        train(0, 1, args, rank=0)
    else:
        if requested_gpus > visible_gpus:
            raise ValueError(f"requested devices={requested_gpus}, but only {visible_gpus} CUDA devices are visible")
        print(f"Detected {visible_gpus} GPUs, launching DDP on {requested_gpus} GPUs with mp.spawn")
        mp.spawn(train, nprocs=requested_gpus, args=(requested_gpus, args))


if __name__ == "__main__":
    main()
