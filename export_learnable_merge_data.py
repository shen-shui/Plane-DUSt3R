import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from learnable_plane_merge.features import wall_info_to_base_targets, wall_info_to_feature
from visualize_plane_merge import (
    closest_pose_index,
    collect_annotation_gt_planes,
    collect_pred_planes,
    estimate_metric_scale,
    extract_annotation_wall_segments,
    load_annotation,
    match_planes,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export supervised wall-candidate tokens for learnable plane merge."
    )
    parser.add_argument("--result_root", required=True, help="Plane-DUSt3R cached result root.")
    parser.add_argument("--gt_root", required=True, help="Structured3D dataset root.")
    parser.add_argument("--output_dir", required=True, help="Directory for exported .npz samples.")
    parser.add_argument("--scene_prefix", default="scene_")
    return parser.parse_args()


def endpoint_target_for_match(gt_name, gt_segments, fallback):
    gt_id = gt_name.split("_")[1]
    segment = gt_segments.get(f"gt_{gt_id}")
    if segment is None:
        return fallback
    _, p0, p1 = segment
    direct = np.asarray([p0[0], p0[1], p1[0], p1[1]], dtype=np.float32)
    flipped = np.asarray([p1[0], p1[1], p0[0], p0[1]], dtype=np.float32)
    return direct if np.linalg.norm(direct - fallback) <= np.linalg.norm(flipped - fallback) else flipped


def orient_plane_target(gt_plane, base_plane):
    gt_plane = np.asarray(gt_plane, dtype=np.float32)
    if np.dot(gt_plane[:3], base_plane[:3]) < 0:
        gt_plane = -gt_plane
    return gt_plane


def export_room(result_dir, gt_root, scene_name, output_dir):
    node_path = result_dir / "node_data.json"
    dust3r_path = result_dir / "dust3r_output.npz"
    if not node_path.exists() or not dust3r_path.exists():
        return False

    gt_room_path = gt_root / scene_name / "2D_rendering" / result_dir.name / "perspective" / "full"
    annotation = load_annotation(gt_root / scene_name / "annotation_3d.json")
    if annotation is None or not gt_room_path.exists():
        return False

    node_info = json.loads(node_path.read_text())
    dust3r_output = np.load(dust3r_path)
    poses = dust3r_output["poses"]
    ref_idx = closest_pose_index(poses)
    scale, rt0 = estimate_metric_scale(poses, gt_room_path, ref_idx)

    pred_planes = collect_pred_planes(node_info, scale)
    gt_planes = collect_annotation_gt_planes(annotation, int(result_dir.name), rt0)
    _, _, pairs = match_planes(pred_planes, gt_planes)
    matched_gt_by_pred = {}
    for pred_idx, gt_idx in pairs:
        pred_name, _ = pred_planes[pred_idx]
        gt_name, _ = gt_planes[gt_idx]
        if pred_name.startswith("wall_pred_") and "_wall" in gt_name:
            matched_gt_by_pred[pred_name] = (gt_name, gt_planes[gt_idx][1])

    gt_segments = {
        name: (name, p0, p1)
        for name, p0, p1 in extract_annotation_wall_segments(annotation, int(result_dir.name), rt0)
    }

    features = []
    keep_target = []
    base_plane = []
    base_endpoints = []
    plane_target = []
    endpoint_target = []
    wall_names = []

    for i, info in enumerate(node_info.get("global_plane_info", [])):
        if not info.get("left_endpoint") or not info.get("right_endpoint"):
            continue
        name = f"wall_pred_{i}"
        feature = wall_info_to_feature(info, scale)
        base_p, base_e = wall_info_to_base_targets(info, scale)
        match = matched_gt_by_pred.get(name)

        features.append(feature)
        base_plane.append(base_p)
        base_endpoints.append(base_e)
        wall_names.append(name)
        if match is None:
            keep_target.append(0.0)
            plane_target.append(base_p)
            endpoint_target.append(base_e)
        else:
            gt_name, gt_plane = match
            keep_target.append(1.0)
            plane_target.append(orient_plane_target(gt_plane, base_p))
            endpoint_target.append(endpoint_target_for_match(gt_name, gt_segments, base_e))

    if not features:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{scene_name}_{result_dir.name}.npz"
    np.savez_compressed(
        output_path,
        features=np.asarray(features, dtype=np.float32),
        keep_target=np.asarray(keep_target, dtype=np.float32),
        base_plane=np.asarray(base_plane, dtype=np.float32),
        base_endpoints=np.asarray(base_endpoints, dtype=np.float32),
        plane_target=np.asarray(plane_target, dtype=np.float32),
        endpoint_target=np.asarray(endpoint_target, dtype=np.float32),
        wall_names=np.asarray(wall_names),
        scene=scene_name,
        room=result_dir.name,
    )
    return True


def main():
    args = parse_args()
    result_root = Path(args.result_root)
    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir)
    exported = 0
    failed = 0

    room_dirs = []
    for scene_dir in sorted(path for path in result_root.iterdir() if path.is_dir()):
        scene_name = scene_dir.name
        if not scene_name.startswith(args.scene_prefix):
            scene_name = f"{args.scene_prefix}{scene_name}"
        for result_dir in sorted(path for path in scene_dir.iterdir() if path.is_dir()):
            room_dirs.append((scene_name, result_dir))

    for scene_name, result_dir in tqdm(room_dirs):
        try:
            if export_room(result_dir, gt_root, scene_name, output_dir):
                exported += 1
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            print(f"failed {scene_name}/{result_dir.name}: {exc}")

    print(f"exported={exported}, failed={failed}, output={output_dir}")


if __name__ == "__main__":
    main()
