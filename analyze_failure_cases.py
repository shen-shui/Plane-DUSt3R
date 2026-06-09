import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_floorplan import evaluate_room
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
        description="Analyze Plane-DUSt3R floor-plan failures by separating detection and merge stages."
    )
    parser.add_argument("--result_root", required=True, help="Evaluation output root.")
    parser.add_argument("--gt_root", required=True, help="Structured3D dataset root.")
    parser.add_argument("--output_csv", default="failure_analysis.csv")
    parser.add_argument("--scene_prefix", default="scene_")
    parser.add_argument("--bad_iou", type=float, default=0.4)
    parser.add_argument("--bad_recall", type=float, default=0.5)
    return parser.parse_args()


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def count_single_view_detections(plane_detection):
    if not plane_detection:
        return {
            "single_view_wall_detections": 0,
            "single_view_floor_detections": 0,
            "single_view_ceiling_detections": 0,
            "views_with_wall": 0,
            "views_with_floor": 0,
            "views_with_ceiling": 0,
        }

    wall_count = 0
    floor_count = 0
    ceiling_count = 0
    views_with_wall = 0
    views_with_floor = 0
    views_with_ceiling = 0

    for detection in plane_detection.values():
        walls = len(detection.get("plane", []))
        floors = len(detection.get("floor", []))
        ceilings = len(detection.get("ceiling", []))
        wall_count += walls
        floor_count += floors
        ceiling_count += ceilings
        views_with_wall += int(walls > 0)
        views_with_floor += int(floors > 0)
        views_with_ceiling += int(ceilings > 0)

    return {
        "single_view_wall_detections": wall_count,
        "single_view_floor_detections": floor_count,
        "single_view_ceiling_detections": ceiling_count,
        "views_with_wall": views_with_wall,
        "views_with_floor": views_with_floor,
        "views_with_ceiling": views_with_ceiling,
    }


def get_annotation_context(result_dir, gt_root, scene_name):
    node_path = result_dir / "node_data.json"
    dust3r_path = result_dir / "dust3r_output.npz"
    if not node_path.exists() or not dust3r_path.exists():
        return None

    gt_room_path = gt_root / scene_name / "2D_rendering" / result_dir.name / "perspective" / "full"
    annotation = load_annotation(gt_root / scene_name / "annotation_3d.json")
    if annotation is None or not gt_room_path.exists():
        return None

    node_info = load_json(node_path)
    dust3r_output = np.load(dust3r_path)
    poses = dust3r_output["poses"]
    ref_idx = closest_pose_index(poses)
    scale, rt0 = estimate_metric_scale(poses, gt_room_path, ref_idx)
    pred_planes = collect_pred_planes(node_info, scale)
    gt_planes = collect_annotation_gt_planes(annotation, int(result_dir.name), rt0)
    matched_pred, matched_gt, pairs = match_planes(pred_planes, gt_planes)
    gt_segments = extract_annotation_wall_segments(annotation, int(result_dir.name), rt0)

    wall_pairs = []
    for pred_idx, gt_idx in pairs:
        pred_name, _ = pred_planes[pred_idx]
        gt_name, _ = gt_planes[gt_idx]
        if pred_name.startswith("wall_pred_") and "_wall" in gt_name:
            wall_pairs.append((pred_idx, gt_idx))

    return {
        "node_info": node_info,
        "pred_planes": pred_planes,
        "gt_planes": gt_planes,
        "gt_wall_segments": gt_segments,
        "matched_planes": len(pairs),
        "matched_walls": len(wall_pairs),
        "matched_pred": len(matched_pred),
        "matched_gt": len(matched_gt),
    }


def infer_failure_type(row):
    gt_walls = row["gt_walls"]
    merged_walls = row["merged_walls"]
    matched_walls = row["matched_walls"]
    single_walls = row["single_view_wall_detections"]
    views = row["views"]
    wall_recall = row["wall_recall"]
    polygon_iou = row["floorplan_polygon_iou"]

    if gt_walls == 0:
        return "no_gt_walls"
    if single_walls == 0:
        return "single_view_detection_failed"
    if views <= 1 and wall_recall < 0.5:
        return "too_few_views"
    if merged_walls == 0:
        return "merge_dropped_all_walls"
    if single_walls >= gt_walls and merged_walls < gt_walls and wall_recall < 0.5:
        return "merge_lost_candidate_walls"
    if merged_walls >= gt_walls and matched_walls < max(1, gt_walls // 2):
        return "global_alignment_or_plane_matching_failed"
    if np.isfinite(polygon_iou) and polygon_iou < 0.4 and wall_recall >= 0.5:
        return "wall_endpoint_or_shape_error"
    if wall_recall < 0.5:
        return "low_wall_recall"
    return "ok_or_minor_error"


def analyze_room(result_dir, gt_root, scene_name):
    metric_row = evaluate_room(result_dir, gt_root, scene_name)
    if metric_row is None:
        return None

    plane_detection = load_json(result_dir / "plane_detection.json")
    detection_counts = count_single_view_detections(plane_detection)
    context = get_annotation_context(result_dir, gt_root, scene_name)
    if context is None:
        return None

    node_info = context["node_info"]
    merged_walls = len(node_info.get("global_plane_info", []))
    merged_floor = int(bool(node_info.get("floor_pparam")))
    merged_ceiling = int(bool(node_info.get("ceiling_pparam")))

    row = {
        **metric_row,
        **detection_counts,
        "merged_walls": merged_walls,
        "merged_floor": merged_floor,
        "merged_ceiling": merged_ceiling,
        "single_to_merged_wall_ratio": (
            merged_walls / detection_counts["single_view_wall_detections"]
            if detection_counts["single_view_wall_detections"]
            else 0
        ),
    }
    row["failure_type_candidate"] = infer_failure_type(row)
    return row


def main():
    args = parse_args()
    result_root = Path(args.result_root)
    gt_root = Path(args.gt_root)
    rows = []

    for scene_dir in sorted(path for path in result_root.iterdir() if path.is_dir()):
        scene_name = scene_dir.name
        if not scene_name.startswith(args.scene_prefix):
            scene_name = f"{args.scene_prefix}{scene_name}"
        for result_dir in sorted(path for path in scene_dir.iterdir() if path.is_dir()):
            row = analyze_room(result_dir, gt_root, scene_name)
            if row is not None:
                rows.append(row)

    fields = [
        "scene",
        "room",
        "views",
        "gt_walls",
        "single_view_wall_detections",
        "views_with_wall",
        "merged_walls",
        "matched_walls",
        "wall_precision",
        "wall_recall",
        "floorplan_polygon_iou",
        "mean_angle_error_deg",
        "mean_offset_error_m",
        "mean_endpoint_error_m",
        "single_to_merged_wall_ratio",
        "single_view_floor_detections",
        "single_view_ceiling_detections",
        "views_with_floor",
        "views_with_ceiling",
        "merged_floor",
        "merged_ceiling",
        "pred_planes",
        "gt_planes",
        "matched_planes",
        "plane_precision",
        "plane_recall",
        "failure_type_candidate",
    ]

    with open(args.output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.output_csv}")
    if rows:
        counts = {}
        for row in rows:
            key = row["failure_type_candidate"]
            counts[key] = counts.get(key, 0) + 1
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"{key}: {count}")


if __name__ == "__main__":
    main()
