import argparse
import csv
import math
from pathlib import Path

import numpy as np

from visualize_plane_merge import (
    closest_pose_index,
    collect_annotation_gt_planes,
    collect_pred_planes,
    estimate_metric_scale,
    extract_annotation_wall_segments,
    extract_pred_wall_segments,
    load_annotation,
    match_planes,
    normalize_plane,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export floor-plan metrics for Plane-DUSt3R merged planes."
    )
    parser.add_argument("--result_root", required=True, help="Evaluation output root.")
    parser.add_argument("--gt_root", required=True, help="Structured3D dataset root.")
    parser.add_argument("--output_csv", default="floorplan_metrics.csv")
    parser.add_argument("--scene_prefix", default="scene_")
    return parser.parse_args()


def plane_angle_error_deg(pred, gt):
    pred = normalize_plane(pred)
    gt = normalize_plane(gt)
    cos_angle = np.dot(pred[:3], gt[:3])
    angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
    return min(angle, 180.0 - angle)


def plane_offset_error(pred, gt):
    pred = normalize_plane(pred)
    gt = normalize_plane(gt)
    return abs(pred[3] - gt[3])


def endpoint_error(pred_seg, gt_seg):
    _, p0, p1 = pred_seg
    _, g0, g1 = gt_seg
    direct = (np.linalg.norm(p0 - g0) + np.linalg.norm(p1 - g1)) / 2
    flipped = (np.linalg.norm(p0 - g1) + np.linalg.norm(p1 - g0)) / 2
    return min(direct, flipped)


def polygon_from_segments(segments):
    points = []
    for _, p0, p1 in segments:
        points.append(tuple(np.asarray(p0, dtype=np.float64)))
        points.append(tuple(np.asarray(p1, dtype=np.float64)))
    if len(points) < 3:
        return None

    unique = []
    for point in points:
        if not any(np.linalg.norm(np.asarray(point) - np.asarray(item)) < 1e-3 for item in unique):
            unique.append(point)
    if len(unique) < 3:
        return None

    pts = np.asarray(unique)
    center = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))
    ordered = pts[order]

    try:
        from shapely.geometry import Polygon
    except ImportError:
        return None

    poly = Polygon(ordered).buffer(0)
    if poly.is_empty or poly.area <= 0:
        return None
    return poly


def polygon_iou(pred_segments, gt_segments):
    pred_poly = polygon_from_segments(pred_segments)
    gt_poly = polygon_from_segments(gt_segments)
    if pred_poly is None or gt_poly is None:
        return float("nan")
    union = pred_poly.union(gt_poly).area
    if union <= 0:
        return float("nan")
    return pred_poly.intersection(gt_poly).area / union


def rasterize_segments(segments, bounds, resolution=256, thickness=2):
    xmin, ymin, xmax, ymax = bounds
    scale = (resolution - 1) / max(xmax - xmin, ymax - ymin, 1e-6)
    mask = np.zeros((resolution, resolution), dtype=bool)

    def to_pixel(point):
        x = int(round((point[0] - xmin) * scale))
        y = int(round((point[1] - ymin) * scale))
        return np.clip(x, 0, resolution - 1), np.clip(y, 0, resolution - 1)

    for _, p0, p1 in segments:
        x0, y0 = to_pixel(np.asarray(p0, dtype=np.float64))
        x1, y1 = to_pixel(np.asarray(p1, dtype=np.float64))
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        xs = np.linspace(x0, x1, steps + 1).round().astype(np.int64)
        ys = np.linspace(y0, y1, steps + 1).round().astype(np.int64)
        mask[ys, xs] = True

    if thickness <= 0:
        return mask
    dilated = mask.copy()
    for dy in range(-thickness, thickness + 1):
        for dx in range(-thickness, thickness + 1):
            if dx * dx + dy * dy > thickness * thickness:
                continue
            y_src0 = max(0, -dy)
            y_src1 = min(resolution, resolution - dy)
            x_src0 = max(0, -dx)
            x_src1 = min(resolution, resolution - dx)
            y_dst0 = max(0, dy)
            y_dst1 = min(resolution, resolution + dy)
            x_dst0 = max(0, dx)
            x_dst1 = min(resolution, resolution + dx)
            dilated[y_dst0:y_dst1, x_dst0:x_dst1] |= mask[y_src0:y_src1, x_src0:x_src1]
    return dilated


def line_map_iou(pred_segments, gt_segments, resolution=256, thickness=2):
    points = [p for _, p0, p1 in pred_segments + gt_segments for p in (p0, p1)]
    if not points:
        return float("nan")
    pts = np.asarray(points, dtype=np.float64)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    pad = max(xmax - xmin, ymax - ymin, 0.5) * 0.1
    bounds = (xmin - pad, ymin - pad, xmax + pad, ymax + pad)
    pred_mask = rasterize_segments(pred_segments, bounds, resolution=resolution, thickness=thickness)
    gt_mask = rasterize_segments(gt_segments, bounds, resolution=resolution, thickness=thickness)
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return float("nan")
    return np.logical_and(pred_mask, gt_mask).sum() / union


def get_view_count(result_dir):
    metric_path = result_dir / "metric_results.txt"
    if not metric_path.exists():
        return 0
    return sum(1 for line in metric_path.read_text().splitlines() if line.startswith("["))


def evaluate_room(result_dir, gt_root, scene_name):
    node_path = result_dir / "node_data.json"
    dust3r_path = result_dir / "dust3r_output.npz"
    if not node_path.exists() or not dust3r_path.exists():
        return None

    import json

    gt_room_path = gt_root / scene_name / "2D_rendering" / result_dir.name / "perspective" / "full"
    annotation = load_annotation(gt_root / scene_name / "annotation_3d.json")
    if annotation is None or not gt_room_path.exists():
        return None

    node_info = json.loads(node_path.read_text())
    dust3r_output = np.load(dust3r_path)
    poses = dust3r_output["poses"]
    ref_idx = closest_pose_index(poses)
    scale, rt0 = estimate_metric_scale(poses, gt_room_path, ref_idx)

    pred_planes = collect_pred_planes(node_info, scale)
    gt_planes = collect_annotation_gt_planes(annotation, int(result_dir.name), rt0)
    matched_pred, matched_gt, pairs = match_planes(pred_planes, gt_planes)

    pred_segments = extract_pred_wall_segments(node_info, scale)
    gt_segments = extract_annotation_wall_segments(annotation, int(result_dir.name), rt0)
    pred_segment_map = {name: item for item in pred_segments for name in [item[0]]}
    gt_segment_map = {name: item for item in gt_segments for name in [item[0]]}

    wall_pairs = []
    angle_errors = []
    offset_errors = []
    endpoint_errors = []
    for pred_idx, gt_idx in pairs:
        pred_name, pred_plane = pred_planes[pred_idx]
        gt_name, gt_plane = gt_planes[gt_idx]
        if not pred_name.startswith("wall_pred_") or "_wall" not in gt_name:
            continue
        wall_pairs.append((pred_idx, gt_idx))
        angle_errors.append(plane_angle_error_deg(pred_plane, gt_plane))
        offset_errors.append(plane_offset_error(pred_plane, gt_plane))

        gt_id = gt_name.split("_")[1]
        pred_seg = pred_segment_map.get(pred_name)
        gt_seg = gt_segment_map.get(f"gt_{gt_id}")
        if pred_seg is not None and gt_seg is not None:
            endpoint_errors.append(endpoint_error(pred_seg, gt_seg))

    pred_wall_count = len(pred_segments)
    gt_wall_count = len(gt_segments)
    matched_wall_count = len(wall_pairs)

    return {
        "scene": scene_name,
        "room": result_dir.name,
        "views": get_view_count(result_dir),
        "pred_planes": len(pred_planes),
        "gt_planes": len(gt_planes),
        "matched_planes": len(pairs),
        "plane_precision": len(matched_pred) / len(pred_planes) if pred_planes else 0,
        "plane_recall": len(matched_gt) / len(gt_planes) if gt_planes else 0,
        "pred_walls": pred_wall_count,
        "gt_walls": gt_wall_count,
        "matched_walls": matched_wall_count,
        "wall_precision": matched_wall_count / pred_wall_count if pred_wall_count else 0,
        "wall_recall": matched_wall_count / gt_wall_count if gt_wall_count else 0,
        "mean_angle_error_deg": np.mean(angle_errors) if angle_errors else float("nan"),
        "mean_offset_error_m": np.mean(offset_errors) if offset_errors else float("nan"),
        "mean_endpoint_error_m": np.mean(endpoint_errors) if endpoint_errors else float("nan"),
        "floorplan_polygon_iou": polygon_iou(pred_segments, gt_segments),
        "floorplan_line_iou": line_map_iou(pred_segments, gt_segments),
    }


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
            row = evaluate_room(result_dir, gt_root, scene_name)
            if row is not None:
                rows.append(row)

    fields = [
        "scene",
        "room",
        "views",
        "pred_planes",
        "gt_planes",
        "matched_planes",
        "plane_precision",
        "plane_recall",
        "pred_walls",
        "gt_walls",
        "matched_walls",
        "wall_precision",
        "wall_recall",
        "mean_angle_error_deg",
        "mean_offset_error_m",
        "mean_endpoint_error_m",
        "floorplan_polygon_iou",
        "floorplan_line_iou",
    ]
    with open(args.output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        print(f"wrote {len(rows)} rows to {args.output_csv}")
        for key in [
            "plane_precision",
            "plane_recall",
            "wall_precision",
            "wall_recall",
            "floorplan_polygon_iou",
            "floorplan_line_iou",
        ]:
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            if len(values):
                print(f"{key}: {values.mean():.4f}")


if __name__ == "__main__":
    main()
