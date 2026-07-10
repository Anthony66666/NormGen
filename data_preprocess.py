#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


CSV_DTYPES = {
    "case_id": float,
    "track_id": int,
    "frame_id": int,
    "timestamp_ms": int,
    "agent_type": str,
    "x": float,
    "y": float,
    "vx": float,
    "vy": float,
    "psi_rad": float,
    "length": float,
    "width": float,
}

NUMERIC_COLUMNS = [
    "case_id",
    "track_id",
    "frame_id",
    "timestamp_ms",
    "x",
    "y",
    "vx",
    "vy",
    "psi_rad",
    "length",
    "width",
]

INTEGER_COLUMNS = ["case_id", "track_id", "frame_id", "timestamp_ms"]

REQUIRED_COLUMNS = [
    "case_id",
    "track_id",
    "frame_id",
    "timestamp_ms",
    "agent_type",
    "x",
    "y",
    "vx",
    "vy",
]

LABEL_MAPPING = {
    "straight": 0,
    "left_turn": 1,
    "right_turn": 2,
    "stationary": 3,
    "unknown": 4,
}

MAP_TYPE_MAPPING = {
    "centerline": 0,
    "boundary": 1,
    "crosswalk": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert INTERACTION multi-agent raw data into MapGlow npz files."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to INTERACTION-Dataset-DR-multi-v1_2 root.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "val"),
        help="Which split to preprocess.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_interaction",
        help="Directory for output npz files.",
    )
    parser.add_argument(
        "--max_agents",
        type=int,
        default=32,
        help="Maximum number of agents per scene.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=40,
        help="Number of frames kept per scene.",
    )
    parser.add_argument(
        "--history_steps",
        type=int,
        default=10,
        help=(
            "Number of leading observed frames used to define the forecasting "
            "coordinate system and context-agent set."
        ),
    )
    parser.add_argument(
        "--forecasting_safe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Derive agent selection, centering, adaptive scale, labels, and map "
            "ranking only from the first --history_steps frames. Disable this "
            "for legacy full-sequence initialization preprocessing."
        ),
    )
    parser.add_argument(
        "--max_lanes",
        type=int,
        default=128,
        help="Maximum number of map polylines per scene.",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=20,
        help="Number of points sampled per map polyline.",
    )
    parser.add_argument(
        "--position_scale",
        type=float,
        default=50.0,
        help="Fixed fallback scale used to normalize x/y and map coordinates.",
    )
    parser.add_argument(
        "--adaptive_position_scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a per-scene scale derived from selected agent extents instead of fixed --position_scale.",
    )
    parser.add_argument(
        "--scale_margin",
        type=float,
        default=1.25,
        help="Multiplier applied to the max centered agent extent when adaptive scaling is enabled.",
    )
    parser.add_argument(
        "--min_position_scale",
        type=float,
        default=50.0,
        help="Lower bound for adaptive per-scene position scale.",
    )
    parser.add_argument(
        "--velocity_scale",
        type=float,
        default=15.0,
        help="Fixed scale used to normalize vx/vy.",
    )
    parser.add_argument(
        "--dimension_scale",
        type=float,
        default=10.0,
        help="Fixed scale used to normalize length/width before saving dimensions.",
    )
    parser.add_argument(
        "--normalize_yaw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize yaw by pi to keep it in roughly [-1, 1].",
    )
    parser.add_argument(
        "--placeholder_label",
        type=int,
        default=4,
        help="Label id used for all valid agents when raw labels are unavailable.",
    )
    parser.add_argument(
        "--compute_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Infer straight/left/right/stationary labels from each selected track.",
    )
    parser.add_argument(
        "--stationary_dist_threshold",
        type=float,
        default=0.0,
        help="Metric displacement below this value is labeled stationary.",
    )
    parser.add_argument(
        "--turn_angle_threshold_deg",
        type=float,
        default=30.0,
        help="Absolute wrapped heading change above this value is labeled left/right turn.",
    )
    parser.add_argument(
        "--agent_sort",
        type=str,
        default="distance",
        choices=("distance", "track_id"),
        help="Ordering rule for agent slots after filtering.",
    )
    parser.add_argument(
        "--require_full_trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require visibility in every reference frame: history frames in "
            "forecasting-safe mode, otherwise all seq_len frames."
        ),
    )
    parser.add_argument(
        "--limit_scenes",
        type=int,
        default=None,
        help="Optional scene limit for debugging.",
    )
    return parser.parse_args()


def load_track_csv(csv_path):
    df = pd.read_csv(csv_path, dtype={"agent_type": "string"})

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    bad_rows = df[REQUIRED_COLUMNS].isna().any(axis=1)
    num_bad_rows = int(bad_rows.sum())
    if num_bad_rows:
        df = df.loc[~bad_rows].copy()

    for column in INTEGER_COLUMNS:
        df[column] = df[column].astype(np.int64)

    df["agent_type"] = df["agent_type"].astype(str)
    return df, num_bad_rows


def resample_polyline(points, num_points):
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return np.zeros((num_points, 2), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points, num_points, axis=0).astype(np.float32)

    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-6:
        return np.repeat(points[:1], num_points, axis=0).astype(np.float32)

    samples = np.linspace(0.0, total, num_points, dtype=np.float32)
    x = np.interp(samples, cum, points[:, 0])
    y = np.interp(samples, cum, points[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def align_polyline_direction(poly_a, poly_b):
    if len(poly_a) < 2 or len(poly_b) < 2:
        return poly_a, poly_b

    same = np.linalg.norm(poly_a[0] - poly_b[0]) + np.linalg.norm(poly_a[-1] - poly_b[-1])
    reverse = np.linalg.norm(poly_a[0] - poly_b[-1]) + np.linalg.norm(poly_a[-1] - poly_b[0])
    if reverse < same:
        poly_b = poly_b[::-1]
    return poly_a, poly_b


def compute_centerline(left_pts, right_pts, num_points=50):
    left_pts, right_pts = align_polyline_direction(np.asarray(left_pts), np.asarray(right_pts))
    left = resample_polyline(left_pts, num_points)
    right = resample_polyline(right_pts, num_points)
    return ((left + right) * 0.5).astype(np.float32)


def parse_speed_limit(sign_type):
    if not sign_type:
        return 0.0
    token = sign_type.strip().lower()
    number = "".join(ch for ch in token if (ch.isdigit() or ch == "."))
    if not number:
        return 0.0
    value = float(number)
    if token.endswith("mph"):
        return value * 0.44704
    if token.endswith("kmh"):
        return value / 3.6
    return value


def parse_osm_xy(map_path):
    tree = ET.parse(map_path)
    root = tree.getroot()

    nodes = {}
    ways = {}
    lanelet_relations = []
    speed_limit_relations = {}

    for node in root.findall("node"):
        nodes[node.attrib["id"]] = np.array(
            [float(node.attrib["x"]), float(node.attrib["y"])],
            dtype=np.float32,
        )

    for way in root.findall("way"):
        node_refs = [nd.attrib["ref"] for nd in way.findall("nd")]
        coords = np.array([nodes[ref] for ref in node_refs if ref in nodes], dtype=np.float32)
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag")}
        ways[way.attrib["id"]] = {
            "coords": coords,
            "tags": tags,
        }

    for rel in root.findall("relation"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in rel.findall("tag")}
        if tags.get("type") == "regulatory_element" and tags.get("subtype") == "speed_limit":
            speed_limit_relations[rel.attrib["id"]] = parse_speed_limit(tags.get("sign_type", ""))
            continue
        if tags.get("type") != "lanelet":
            continue
        members = defaultdict(list)
        for member in rel.findall("member"):
            members[member.attrib.get("role", "")].append(member.attrib["ref"])
        lanelet_relations.append(
            {
                "id": rel.attrib["id"],
                "left": members.get("left", []),
                "right": members.get("right", []),
                "subtype": tags.get("subtype", ""),
                "regulatory_elements": members.get("regulatory_element", []),
            }
        )

    return ways, lanelet_relations, speed_limit_relations


def build_raw_map_polylines(map_path):
    ways, lanelet_relations, speed_limit_relations = parse_osm_xy(map_path)
    polylines = []
    used_way_ids = {}

    for rel in lanelet_relations:
        if not rel["left"] or not rel["right"]:
            continue
        left_way = ways.get(rel["left"][0])
        right_way = ways.get(rel["right"][0])
        if left_way is None or right_way is None:
            continue

        left_pts = left_way["coords"]
        right_pts = right_way["coords"]
        if len(left_pts) < 2 or len(right_pts) < 2:
            continue

        speed_limit = 0.0
        for reg_id in rel.get("regulatory_elements", []):
            if reg_id in speed_limit_relations:
                speed_limit = speed_limit_relations[reg_id]
                break

        centerline = compute_centerline(left_pts, right_pts)
        polylines.append(
            {
                "coords": centerline,
                "type": MAP_TYPE_MAPPING["centerline"],
                "speed_limit_mps": np.float32(speed_limit),
            }
        )

        for way_id in (rel["left"][0], rel["right"][0]):
            if way_id not in used_way_ids:
                coords = ways[way_id]["coords"]
                if len(coords) >= 2:
                    polylines.append(
                        {
                            "coords": coords.astype(np.float32),
                            "type": MAP_TYPE_MAPPING["boundary"],
                            "speed_limit_mps": np.float32(speed_limit),
                        }
                    )
                used_way_ids[way_id] = speed_limit

    for way_id, way in ways.items():
        tags = way["tags"]
        if tags.get("type") == "pedestrian_marking":
            coords = way["coords"]
            if len(coords) >= 2:
                polylines.append(
                    {
                        "coords": coords.astype(np.float32),
                        "type": MAP_TYPE_MAPPING["crosswalk"],
                        "speed_limit_mps": np.float32(0.0),
                    }
                )

    deduped = []
    seen = set()
    for poly in polylines:
        coords = poly["coords"]
        if len(coords) < 2:
            continue
        key = (
            poly["type"],
            tuple(np.round(coords[[0, -1]].reshape(-1), 3).tolist()),
        )
        if key in seen:
            continue
        deduped.append(poly)
        seen.add(key)
    return deduped


def map_agent_type(agent_type):
    if agent_type == "car":
        return 0
    return 1


def fill_agent_dimensions(agent_type, lengths, widths):
    lengths = lengths.astype(np.float32, copy=True)
    widths = widths.astype(np.float32, copy=True)

    if agent_type == "car":
        lengths[~np.isfinite(lengths)] = 0.0
        widths[~np.isfinite(widths)] = 0.0
    else:
        lengths[:] = 0.5
        widths[:] = 0.5

    return lengths, widths


def infer_yaw(track_rows):
    psi = track_rows["psi_rad"].to_numpy(dtype=np.float32)
    vx = track_rows["vx"].to_numpy(dtype=np.float32)
    vy = track_rows["vy"].to_numpy(dtype=np.float32)
    missing = ~np.isfinite(psi)
    if missing.any():
        derived = np.arctan2(vy, vx).astype(np.float32)
        psi[missing] = derived[missing]
    psi[~np.isfinite(psi)] = 0.0
    return psi


def scene_center_from_tracks(case_df, frame_ids, track_ids):
    scene_rows = case_df[
        case_df["frame_id"].isin(frame_ids) & case_df["track_id"].isin(track_ids)
    ]
    if len(scene_rows) == 0:
        return np.zeros(2, dtype=np.float32)
    xy = scene_rows[["x", "y"]].to_numpy(dtype=np.float32)
    return np.nanmean(xy, axis=0).astype(np.float32)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def infer_track_label(track_rows, psi, args):
    xy = track_rows[["x", "y"]].to_numpy(dtype=np.float32)
    valid_xy = np.isfinite(xy).all(axis=1)
    if valid_xy.sum() < 2:
        return LABEL_MAPPING["unknown"]

    valid_idx = np.flatnonzero(valid_xy)
    displacement = float(np.linalg.norm(xy[valid_idx[-1]] - xy[valid_idx[0]]))
    if args.stationary_dist_threshold > 0.0 and displacement < args.stationary_dist_threshold:
        return LABEL_MAPPING["stationary"]

    psi = np.asarray(psi, dtype=np.float32)
    valid_psi = np.isfinite(psi)
    if valid_psi.sum() < 2:
        return LABEL_MAPPING["unknown"]

    psi_unwrapped = np.unwrap(psi[valid_psi].astype(np.float64))
    heading_delta = float(wrap_to_pi(psi_unwrapped[-1] - psi_unwrapped[0]))
    turn_threshold = math.radians(float(args.turn_angle_threshold_deg))

    if heading_delta > turn_threshold:
        return LABEL_MAPPING["left_turn"]
    if heading_delta < -turn_threshold:
        return LABEL_MAPPING["right_turn"]
    return LABEL_MAPPING["straight"]


def track_distance_to_center(track_df, frame_ids, center):
    track_rows = track_df[track_df["frame_id"].isin(frame_ids)]
    if len(track_rows) == 0:
        return float("inf")

    xy = track_rows[["x", "y"]].to_numpy(dtype=np.float32)
    valid_xy = np.isfinite(xy).all(axis=1)
    if not valid_xy.any():
        return float("inf")

    distances = np.linalg.norm(xy[valid_xy] - center[None, :], axis=1)
    return float(np.mean(distances))


def compute_scene_scale(case_df, frame_ids, selected_track_ids, center, args):
    if not args.adaptive_position_scale:
        return float(args.position_scale)

    scene_rows = case_df[
        case_df["frame_id"].isin(frame_ids) & case_df["track_id"].isin(selected_track_ids)
    ]
    if len(scene_rows) == 0:
        return float(args.position_scale)

    xy = scene_rows[["x", "y"]].to_numpy(dtype=np.float32)
    valid_xy = np.isfinite(xy).all(axis=1)
    if not valid_xy.any():
        return float(args.position_scale)

    centered_xy = xy[valid_xy] - center[None, :]
    max_extent = float(np.max(np.abs(centered_xy)))
    if not np.isfinite(max_extent) or max_extent <= 0.0:
        return float(args.position_scale)

    scale = max(
        max_extent * float(args.scale_margin),
        float(args.min_position_scale),
    )
    return float(scale)


def build_scene_tensor(
    case_df,
    map_name,
    raw_map_polylines,
    args,
):
    frame_ids = sorted(case_df["frame_id"].unique().tolist())
    if len(frame_ids) < args.seq_len:
        return None, f"too_few_frames:{len(frame_ids)}"
    frame_ids = frame_ids[: args.seq_len]
    frame_to_idx = {frame_id: idx for idx, frame_id in enumerate(frame_ids)}

    history_steps = int(getattr(args, "history_steps", min(10, args.seq_len)))
    if history_steps <= 0 or history_steps > args.seq_len:
        raise ValueError(
            "history_steps must be in [1, seq_len], got "
            f"history_steps={history_steps}, seq_len={args.seq_len}"
        )
    forecasting_safe = bool(getattr(args, "forecasting_safe", False))
    if forecasting_safe and history_steps >= args.seq_len:
        raise ValueError(
            "forecasting-safe preprocessing requires at least one future frame: "
            f"history_steps={history_steps}, seq_len={args.seq_len}"
        )
    reference_frame_ids = frame_ids[:history_steps] if forecasting_safe else frame_ids
    reference_frame_count = len(reference_frame_ids)

    candidate_tracks = []
    track_groups = {}
    for track_id, track_df in case_df.groupby("track_id"):
        track_groups[track_id] = track_df
        frame_set = set(track_df["frame_id"].tolist())
        # In forecasting-safe mode, even the candidate set and its ordering must
        # be reproducible at inference time, before any future row is observed.
        covered = sum(frame_id in frame_set for frame_id in reference_frame_ids)
        if args.require_full_trajectory and covered != reference_frame_count:
            continue
        if not args.require_full_trajectory and covered == 0:
            continue
        candidate_tracks.append((track_id, covered))

    if not candidate_tracks:
        return None, "no_valid_tracks"

    candidate_track_ids = [track_id for track_id, _ in candidate_tracks]
    center = scene_center_from_tracks(case_df, reference_frame_ids, candidate_track_ids)

    if args.agent_sort == "distance":
        candidate_tracks.sort(
            key=lambda item: (
                -item[1],
                track_distance_to_center(
                    track_groups[item[0]], reference_frame_ids, center
                ),
                item[0],
            )
        )
    else:
        candidate_tracks.sort(key=lambda item: (-item[1], item[0]))

    selected_track_ids = [track_id for track_id, _ in candidate_tracks[: args.max_agents]]
    center = scene_center_from_tracks(case_df, reference_frame_ids, selected_track_ids)
    position_scale = compute_scene_scale(
        case_df, reference_frame_ids, selected_track_ids, center, args
    )

    traj = np.zeros((5, args.seq_len, args.max_agents), dtype=np.float32)
    dimensions = np.zeros((2, args.seq_len, args.max_agents), dtype=np.float32)
    timestep_mask = np.zeros((args.seq_len, args.max_agents), dtype=bool)
    labels = np.full((args.max_agents,), args.placeholder_label, dtype=np.int64)
    agent_types = np.zeros((args.max_agents,), dtype=np.int64)
    track_ids = np.full((args.max_agents,), -1, dtype=np.int64)

    for agent_slot, track_id in enumerate(selected_track_ids):
        track_df = track_groups[track_id].copy()
        track_df = track_df[track_df["frame_id"].isin(frame_ids)].sort_values("frame_id")
        raw_agent_type = str(track_df["agent_type"].iloc[0])
        raw_psi = infer_yaw(track_df)
        x = ((track_df["x"].to_numpy(dtype=np.float32) - center[0]) / position_scale).astype(np.float32)
        y = ((track_df["y"].to_numpy(dtype=np.float32) - center[1]) / position_scale).astype(np.float32)
        vx = (track_df["vx"].to_numpy(dtype=np.float32) / args.velocity_scale).astype(np.float32)
        vy = (track_df["vy"].to_numpy(dtype=np.float32) / args.velocity_scale).astype(np.float32)
        lengths = track_df["length"].to_numpy(dtype=np.float32)
        widths = track_df["width"].to_numpy(dtype=np.float32)
        lengths, widths = fill_agent_dimensions(raw_agent_type, lengths, widths)
        lengths = (lengths / args.dimension_scale).astype(np.float32)
        widths = (widths / args.dimension_scale).astype(np.float32)
        if args.normalize_yaw:
            psi = (raw_psi / np.pi).astype(np.float32)
        else:
            psi = raw_psi.astype(np.float32)

        for row_idx, frame_id in enumerate(track_df["frame_id"].tolist()):
            t = frame_to_idx[frame_id]
            traj[0, t, agent_slot] = x[row_idx]
            traj[1, t, agent_slot] = y[row_idx]
            traj[2, t, agent_slot] = vx[row_idx]
            traj[3, t, agent_slot] = vy[row_idx]
            traj[4, t, agent_slot] = psi[row_idx]
            dimensions[0, t, agent_slot] = lengths[row_idx]
            dimensions[1, t, agent_slot] = widths[row_idx]
            timestep_mask[t, agent_slot] = True

        if args.compute_labels:
            if forecasting_safe:
                label_rows = track_df[
                    track_df["frame_id"].isin(reference_frame_ids)
                ].copy()
                label_psi = infer_yaw(label_rows)
            else:
                label_rows = track_df
                label_psi = raw_psi
            labels[agent_slot] = infer_track_label(label_rows, label_psi, args)
        else:
            labels[agent_slot] = args.placeholder_label
        agent_types[agent_slot] = map_agent_type(raw_agent_type)
        track_ids[agent_slot] = int(track_id)

    map_data = np.zeros((args.max_lanes, args.num_points, 2), dtype=np.float32)
    map_mask = np.zeros((args.max_lanes, args.num_points), dtype=bool)
    map_type = np.zeros((args.max_lanes,), dtype=np.int64)
    map_speed_limit = np.zeros((args.max_lanes,), dtype=np.float32)

    if raw_map_polylines:
        centerline_ranked = []
        other_ranked = []
        for poly_meta in raw_map_polylines:
            poly = poly_meta["coords"]
            dist = np.linalg.norm(poly - center[None, :], axis=1).min()
            item = (float(dist), poly_meta)
            if int(poly_meta["type"]) == MAP_TYPE_MAPPING["centerline"]:
                centerline_ranked.append(item)
            else:
                other_ranked.append(item)

        centerline_ranked.sort(key=lambda item: item[0])
        other_ranked.sort(key=lambda item: item[0])
        ranked = centerline_ranked + other_ranked

        for lane_idx, (_, poly_meta) in enumerate(ranked[: args.max_lanes]):
            poly = poly_meta["coords"]
            sampled = resample_polyline(poly, args.num_points)
            sampled = ((sampled - center[None, :]) / position_scale).astype(np.float32)
            map_data[lane_idx] = sampled
            map_mask[lane_idx] = True
            map_type[lane_idx] = int(poly_meta["type"])
            map_speed_limit[lane_idx] = np.float32(poly_meta["speed_limit_mps"])

    scene_stats = {
        "mean": center.astype(np.float32),
        "scale": np.float32(position_scale),
        "reference": "history" if forecasting_safe else "full_sequence",
        "history_steps": np.int64(history_steps),
    }
    history_timestep_mask = timestep_mask[:history_steps].copy()
    future_timestep_mask = timestep_mask[history_steps:].copy()
    if forecasting_safe:
        context_agent_mask = history_timestep_mask.any(axis=0)
    else:
        context_agent_mask = timestep_mask.any(axis=0)
    future_vehicle_mask = future_timestep_mask.any(axis=0)
    return {
        "trajectories": traj,
        "dimensions": dimensions,
        "timestep_mask": timestep_mask,
        "history_timestep_mask": history_timestep_mask,
        "future_timestep_mask": future_timestep_mask,
        "vehicle_mask": context_agent_mask.copy(),
        "context_agent_mask": context_agent_mask,
        "future_vehicle_mask": future_vehicle_mask,
        "labels": labels,
        "agent_types": agent_types,
        "track_ids": track_ids,
        "map_name": map_name,
        "scene_stats": scene_stats,
        "map_data": map_data,
        "map_mask": map_mask,
        "map_type": map_type,
        "map_speed_limit": map_speed_limit,
    }, None


def collect_csv_files(root, split):
    split_dir = Path(root) / split
    suffix = f"_{split}.csv"
    return sorted(split_dir.glob(f"*{suffix}"))


def main():
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = collect_csv_files(root, args.split)
    if not csv_files:
        raise FileNotFoundError(f"No csv files found under {root / args.split}")

    map_dir = root / "maps"
    map_cache = {}

    trajectories = []
    timestep_masks = []
    history_timestep_masks = []
    future_timestep_masks = []
    vehicle_masks = []
    context_agent_masks = []
    future_vehicle_masks = []
    labels = []
    agent_types = []
    track_ids = []
    map_names = []
    scene_stats = []
    dimensions = []
    map_data_all = []
    map_mask_all = []
    map_type_all = []
    map_speed_limit_all = []
    scene_ids = []

    skipped = defaultdict(int)
    scene_count = 0

    for csv_path in csv_files:
        map_name = csv_path.name.replace(f"_{args.split}.csv", "")
        map_path = map_dir / f"{map_name}.osm_xy"
        if not map_path.exists():
            skipped["missing_map"] += 1
            print(f"[skip] missing map: {map_path}")
            continue

        if map_name not in map_cache:
            map_cache[map_name] = build_raw_map_polylines(map_path)

        print(f"[load] {csv_path.name}")
        df, bad_csv_rows = load_track_csv(csv_path)
        if bad_csv_rows:
            skipped["bad_csv_rows"] += bad_csv_rows
            print(f"[skip] bad csv rows in {csv_path.name}: {bad_csv_rows}")

        for case_id, case_df in df.groupby("case_id", sort=True):
            scene, reason = build_scene_tensor(case_df, map_name, map_cache[map_name], args)
            if scene is None:
                skipped[reason] += 1
                continue

            trajectories.append(scene["trajectories"])
            dimensions.append(scene["dimensions"])
            timestep_masks.append(scene["timestep_mask"])
            history_timestep_masks.append(scene["history_timestep_mask"])
            future_timestep_masks.append(scene["future_timestep_mask"])
            vehicle_masks.append(scene["vehicle_mask"])
            context_agent_masks.append(scene["context_agent_mask"])
            future_vehicle_masks.append(scene["future_vehicle_mask"])
            labels.append(scene["labels"])
            agent_types.append(scene["agent_types"])
            track_ids.append(scene["track_ids"])
            map_names.append(scene["map_name"])
            scene_stats.append(scene["scene_stats"])
            map_data_all.append(scene["map_data"])
            map_mask_all.append(scene["map_mask"])
            map_type_all.append(scene["map_type"])
            map_speed_limit_all.append(scene["map_speed_limit"])
            scene_ids.append(case_id)
            scene_count += 1

            if args.limit_scenes is not None and scene_count >= args.limit_scenes:
                break

        if args.limit_scenes is not None and scene_count >= args.limit_scenes:
            break

    if not trajectories:
        raise RuntimeError("No scenes were generated. Check split, filters, and input paths.")

    trajectories = np.stack(trajectories, axis=0).astype(np.float32)
    dimensions = np.stack(dimensions, axis=0).astype(np.float32)
    timestep_masks = np.stack(timestep_masks, axis=0).astype(bool)
    history_timestep_masks = np.stack(history_timestep_masks, axis=0).astype(bool)
    future_timestep_masks = np.stack(future_timestep_masks, axis=0).astype(bool)
    vehicle_masks = np.stack(vehicle_masks, axis=0).astype(bool)
    context_agent_masks = np.stack(context_agent_masks, axis=0).astype(bool)
    future_vehicle_masks = np.stack(future_vehicle_masks, axis=0).astype(bool)
    labels = np.stack(labels, axis=0).astype(np.int64)
    agent_types = np.stack(agent_types, axis=0).astype(np.int64)
    track_ids = np.stack(track_ids, axis=0).astype(np.int64)
    map_data_all = np.stack(map_data_all, axis=0).astype(np.float32)
    map_mask_all = np.stack(map_mask_all, axis=0).astype(bool)
    map_type_all = np.stack(map_type_all, axis=0).astype(np.int64)
    map_speed_limit_all = np.stack(map_speed_limit_all, axis=0).astype(np.float32)
    map_names = np.asarray(map_names)
    scene_stats = np.asarray(scene_stats, dtype=object)
    scene_ids = np.asarray(scene_ids, dtype=np.int64)
    normalization_center = np.stack(
        [np.asarray(stat["mean"], dtype=np.float32) for stat in scene_stats], axis=0
    ).astype(np.float32)
    scene_scales = np.asarray([float(stat["scale"]) for stat in scene_stats], dtype=np.float32)
    valid_agent_mask = context_agent_masks
    valid_labels = labels[valid_agent_mask]
    label_values, label_counts = np.unique(valid_labels, return_counts=True)

    combined_out = output_dir / f"interaction_multi_{args.split}_combined.npz"

    np.savez_compressed(
        combined_out,
        trajectories=trajectories,
        dimensions=dimensions,
        timestep_mask=timestep_masks,
        history_timestep_mask=history_timestep_masks,
        future_timestep_mask=future_timestep_masks,
        vehicle_mask=vehicle_masks,
        context_agent_mask=context_agent_masks,
        future_vehicle_mask=future_vehicle_masks,
        labels=labels,
        agent_types=agent_types,
        track_ids=track_ids,
        map_names=map_names,
        scene_stats=scene_stats,
        normalization_center=normalization_center,
        normalization_scale=scene_scales,
        normalization_velocity_scale=np.float32(args.velocity_scale),
        normalization_dimension_scale=np.float32(args.dimension_scale),
        normalization_yaw_scale=np.float32(np.pi if args.normalize_yaw else 1.0),
        normalization_metadata_version=np.int64(1),
        history_steps=np.int64(args.history_steps),
        forecasting_safe=np.bool_(args.forecasting_safe),
        label_mapping=np.asarray(LABEL_MAPPING, dtype=object),
        case_ids=scene_ids,
        map_data=map_data_all,
        map_mask=map_mask_all,
        map_type=map_type_all,
        map_speed_limit=map_speed_limit_all,
        map_type_mapping=np.asarray(MAP_TYPE_MAPPING, dtype=object),
        max_lanes=np.int64(args.max_lanes),
        max_points=np.int64(args.num_points),
    )

    summary = {
        "split": args.split,
        "num_scenes": int(len(trajectories)),
        "trajectories_shape": tuple(trajectories.shape),
        "dimensions_shape": tuple(dimensions.shape),
        "timestep_mask_shape": tuple(timestep_masks.shape),
        "history_timestep_mask_shape": tuple(history_timestep_masks.shape),
        "future_timestep_mask_shape": tuple(future_timestep_masks.shape),
        "context_agent_mask_shape": tuple(context_agent_masks.shape),
        "map_shape": tuple(map_data_all.shape),
        "map_type_shape": tuple(map_type_all.shape),
        "map_speed_limit_shape": tuple(map_speed_limit_all.shape),
        "position_scale": float(args.position_scale),
        "adaptive_position_scale": bool(args.adaptive_position_scale),
        "scale_margin": float(args.scale_margin),
        "min_position_scale": float(args.min_position_scale),
        "scene_scale_min": float(scene_scales.min()),
        "scene_scale_mean": float(scene_scales.mean()),
        "scene_scale_max": float(scene_scales.max()),
        "velocity_scale": float(args.velocity_scale),
        "dimension_scale": float(args.dimension_scale),
        "normalize_yaw": bool(args.normalize_yaw),
        "history_steps": int(args.history_steps),
        "forecasting_safe": bool(args.forecasting_safe),
        "normalization_metadata_version": 1,
        "placeholder_label": int(args.placeholder_label),
        "compute_labels": bool(args.compute_labels),
        "stationary_dist_threshold": float(args.stationary_dist_threshold),
        "turn_angle_threshold_deg": float(args.turn_angle_threshold_deg),
        "agent_sort": args.agent_sort,
        "label_counts": {
            str(int(label)): int(count)
            for label, count in zip(label_values, label_counts)
        },
        "require_full_trajectory": bool(args.require_full_trajectory),
        "skipped": dict(skipped),
    }
    summary_path = output_dir / f"interaction_multi_{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[done] combined npz: {combined_out}")
    print(f"[done] summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
