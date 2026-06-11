import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import parse_camera_info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize merged Plane-DUSt3R planes against Structured3D GT planes."
    )
    parser.add_argument(
        "--result_root",
        type=str,
        required=True,
        help="Evaluation output root, e.g. eval_results_sample5.",
    )
    parser.add_argument(
        "--gt_root",
        type=str,
        required=True,
        help="Structured3D root that contains scene_xxxxx/2D_rendering.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="plane_merge_gt_pred.png",
        help="Output image name saved in each room result directory.",
    )
    parser.add_argument(
        "--floorplan_output_name",
        type=str,
        default="floorplan_gt_pred.png",
        help="Floor-plan-like output image saved in each room result directory.",
    )
    parser.add_argument(
        "--annotation_floorplan_output_name",
        type=str,
        default="floorplan_annotation_gt_pred.png",
        help="Floor-plan output using scene annotation_3d.json room semantics as GT.",
    )
    parser.add_argument(
        "--scene_prefix",
        type=str,
        default="scene_",
        help="Prefix used by Structured3D scene folders.",
    )
    return parser.parse_args()


def closest_pose_index(poses):
    identity = np.eye(3)
    best_idx = 0
    best_rot = float("inf")
    best_trans = float("inf")
    for i, pose in enumerate(poses):
        rot_diff = np.linalg.norm(pose[:3, :3] - identity)
        trans_diff = np.linalg.norm(pose[:3, 3])
        if rot_diff < best_rot or (rot_diff == best_rot and trans_diff < best_trans):
            best_idx = i
            best_rot = rot_diff
            best_trans = trans_diff
    return best_idx


def estimate_metric_scale(poses, gt_room_path, ref_idx):
    scales = []
    pose0 = poses[ref_idx]
    rt0, _ = parse_camera_info(
        np.loadtxt(gt_room_path / str(ref_idx) / "camera_pose.txt"), 720, 1280
    )

    for i, pose in enumerate(poses):
        if i == ref_idx:
            continue
        camera_path = gt_room_path / str(i) / "camera_pose.txt"
        if not camera_path.exists():
            continue

        rt, _ = parse_camera_info(np.loadtxt(camera_path), 720, 1280)
        rt_pred = np.linalg.inv(pose) @ pose0
        rt_gt = np.linalg.inv(rt) @ rt0
        t_pred = rt_pred[:3, 3]
        t_gt = rt_gt[:3, 3]
        if np.linalg.norm(t_pred) == 0:
            continue
        scales.append(np.linalg.norm(t_gt) / np.linalg.norm(t_pred))

    return (np.mean(scales) if scales else 11), rt0


def normalize_plane(plane):
    plane = np.asarray(plane, dtype=np.float64).copy()
    norm = np.linalg.norm(plane[:3])
    if norm > 0:
        plane /= norm
    return plane


def is_matched(plane_param1, plane_param2, angle_threshold_deg=10, offset_threshold=0.15):
    normal1, offset1 = plane_param1[:3], plane_param1[3]
    normal2, offset2 = plane_param2[:3], plane_param2[3]
    cos_angle = np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2))
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    angle_deg = min(angle_deg, 180 - angle_deg)
    offset_diff = abs(offset1 - offset2)
    return angle_deg < angle_threshold_deg and offset_diff < offset_threshold


def extract_scene_planes(gt_room_path, rt0):
    plane_dict = {}
    for position_id in os.listdir(gt_room_path):
        view_path = gt_room_path / position_id
        layout_path = view_path / "layout.json"
        camera_path = view_path / "camera_pose.txt"
        if not layout_path.exists() or not camera_path.exists():
            continue
        layout_info = json.loads(layout_path.read_text())
        rt, _ = parse_camera_info(np.loadtxt(camera_path), 720, 1280)
        for plane in layout_info["planes"]:
            plane_id = plane["ID"]
            if plane_id in plane_dict:
                continue
            pparam = np.concatenate([plane["normal"], [plane["offset"]]]).astype(np.float64)
            pparam[-1] /= 1000
            transform_matrix = np.linalg.inv(rt0) @ rt
            plane_dict[plane_id] = pparam @ np.linalg.inv(transform_matrix)
    return plane_dict


def collect_pred_planes(node_info, scale):
    pred = []
    if node_info.get("floor_pparam"):
        plane = np.asarray(node_info["floor_pparam"], dtype=np.float64).copy()
        plane[3] *= scale
        pred.append(("floor", plane))
    if node_info.get("ceiling_pparam"):
        plane = np.asarray(node_info["ceiling_pparam"], dtype=np.float64).copy()
        plane[3] *= scale
        pred.append(("ceiling", plane))
    for i, info in enumerate(node_info.get("global_plane_info", [])):
        plane = np.asarray(info["pparam"], dtype=np.float64).copy()
        plane[3] *= scale
        pred.append((f"wall_pred_{i}", plane))
    return pred


def is_wall(plane):
    return abs(normalize_plane(plane)[1]) < 0.5


def plane_to_xz_line(plane):
    plane = normalize_plane(plane)
    return np.array([plane[0], plane[2], plane[3]], dtype=np.float64)


def intersect_lines(line_a, line_b):
    a1, b1, c1 = line_a
    a2, b2, c2 = line_b
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-8:
        return None
    x = (b1 * c2 - b2 * c1) / det
    z = (c1 * a2 - c2 * a1) / det
    return np.array([x, z])


def line_segment_in_bbox(line, xmin, xmax, zmin, zmax):
    a, b, c = line
    points = []
    if abs(b) > 1e-8:
        for x in (xmin, xmax):
            z = -(a * x + c) / b
            if zmin <= z <= zmax:
                points.append((x, z))
    if abs(a) > 1e-8:
        for z in (zmin, zmax):
            x = -(b * z + c) / a
            if xmin <= x <= xmax:
                points.append((x, z))

    unique = []
    for p in points:
        if not any(np.linalg.norm(np.asarray(p) - np.asarray(q)) < 1e-6 for q in unique):
            unique.append(p)
    if len(unique) < 2:
        return None
    return unique[0], unique[1]


def match_planes(pred_planes, gt_planes):
    matched_pred = set()
    matched_gt = set()
    pairs = []
    for pi, (_, pred) in enumerate(pred_planes):
        for gi, (_, gt) in enumerate(gt_planes):
            if gi in matched_gt:
                continue
            if is_matched(pred, gt):
                matched_pred.add(pi)
                matched_gt.add(gi)
                pairs.append((pi, gi))
                break
    return matched_pred, matched_gt, pairs


def backproject_to_plane(pixel_xy, plane, k):
    plane = np.asarray(plane, dtype=np.float64)
    ray = np.linalg.inv(k) @ np.array([pixel_xy[0], pixel_xy[1], 1.0], dtype=np.float64)
    denom = np.dot(plane[:3], ray)
    if abs(denom) < 1e-8:
        return None
    point = ray * (-plane[3] / denom)
    if not np.all(np.isfinite(point)):
        return None
    return point


def extract_gt_wall_segments(gt_room_path, rt0):
    segments = {}
    for position_id in os.listdir(gt_room_path):
        view_path = gt_room_path / position_id
        layout_path = view_path / "layout.json"
        camera_path = view_path / "camera_pose.txt"
        if not layout_path.exists() or not camera_path.exists():
            continue

        layout_info = json.loads(layout_path.read_text())
        rt, k = parse_camera_info(np.loadtxt(camera_path), 720, 1280)
        transform_matrix = np.linalg.inv(rt0) @ rt
        junctions = [j["coordinate"] for j in layout_info["junctions"]]

        for plane in layout_info["planes"]:
            if plane.get("type") != "wall" or not plane.get("visible_mask"):
                continue
            plane_id = plane["ID"]
            pparam = np.concatenate([plane["normal"], [plane["offset"]]]).astype(np.float64)
            pparam[-1] /= 1000
            points = segments.setdefault(plane_id, [])
            for mask in plane["visible_mask"]:
                for junction_id in mask:
                    point = backproject_to_plane(junctions[junction_id], pparam, k)
                    if point is None:
                        continue
                    point_h = np.r_[point, 1.0]
                    point_ref = transform_matrix @ point_h
                    points.append(point_ref[:3])

    wall_segments = []
    for plane_id, points in segments.items():
        if len(points) < 2:
            continue
        pts = np.asarray(points, dtype=np.float64)[:, [0, 2]]
        center = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
        direction = vh[0]
        values = (pts - center) @ direction
        p0 = center + direction * values.min()
        p1 = center + direction * values.max()
        if np.linalg.norm(p1 - p0) > 1e-4:
            wall_segments.append((f"gt_{plane_id}", p0, p1))
    return wall_segments


def extract_pred_wall_segments(node_info, scale):
    segments = []
    for i, info in enumerate(node_info.get("global_plane_info", [])):
        left = info.get("left_endpoint")
        right = info.get("right_endpoint")
        if not left or not right:
            continue
        p0 = np.asarray(left, dtype=np.float64)[[0, 2]] * scale
        p1 = np.asarray(right, dtype=np.float64)[[0, 2]] * scale
        if np.linalg.norm(p1 - p0) > 1e-4:
            segments.append((f"wall_pred_{i}", p0, p1))
    return segments


def load_annotation(annotation_path):
    if not annotation_path.exists():
        return None
    data = json.loads(annotation_path.read_text())
    return {
        "raw": data,
        "planes": {plane["ID"]: plane for plane in data.get("planes", [])},
        "semantics": {semantic["ID"]: semantic for semantic in data.get("semantics", [])},
    }


def annotation_room_plane_ids(annotation, room_id):
    semantic = annotation["semantics"].get(int(room_id))
    if not semantic:
        return []
    return semantic.get("planeID", [])


def annotation_plane_to_ref(plane, rt0):
    pparam = np.concatenate([plane["normal"], [plane["offset"]]]).astype(np.float64)
    pparam[-1] /= 1000
    return pparam @ rt0


def collect_annotation_gt_planes(annotation, room_id, rt0):
    gt_planes = []
    for plane_id in annotation_room_plane_ids(annotation, room_id):
        plane = annotation["planes"].get(plane_id)
        if not plane:
            continue
        pparam = annotation_plane_to_ref(plane, rt0)
        label = plane.get("type", "plane")
        gt_planes.append((f"gt_{plane_id}_{label}", pparam))
    return gt_planes


def extract_annotation_wall_segments(annotation, room_id, rt0):
    raw = annotation["raw"]
    planes = annotation["planes"]
    junctions = {
        junction["ID"]: np.asarray(junction["coordinate"], dtype=np.float64) / 1000
        for junction in raw.get("junctions", [])
    }
    plane_line = raw.get("planeLineMatrix", [])
    line_junction = raw.get("lineJunctionMatrix", [])
    segments = []
    world_to_ref = np.linalg.inv(rt0)

    for plane_id in annotation_room_plane_ids(annotation, room_id):
        plane = planes.get(plane_id)
        if not plane or plane.get("type") != "wall":
            continue
        if plane_id >= len(plane_line):
            continue

        points = []
        for line_id, linked in enumerate(plane_line[plane_id]):
            if not linked or line_id >= len(line_junction):
                continue
            for junction_id, has_junction in enumerate(line_junction[line_id]):
                if has_junction and junction_id in junctions:
                    point_ref = world_to_ref @ np.r_[junctions[junction_id], 1.0]
                    points.append(point_ref[:3])

        if len(points) < 2:
            continue
        pts = np.asarray(points, dtype=np.float64)[:, [0, 2]]
        center = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
        direction = vh[0]
        values = (pts - center) @ direction
        p0 = center + direction * values.min()
        p1 = center + direction * values.max()
        if np.linalg.norm(p1 - p0) > 1e-4:
            segments.append((f"gt_{plane_id}", p0, p1))
    return segments


def plot_annotation_floorplan(
    result_dir,
    annotation,
    node_info,
    rt0,
    scale,
    pred_planes,
    output_name,
):
    room_id = int(result_dir.name)
    gt_planes = collect_annotation_gt_planes(annotation, room_id, rt0)
    matched_pred, matched_gt, _ = match_planes(pred_planes, gt_planes)
    gt_segments = extract_annotation_wall_segments(annotation, room_id, rt0)
    pred_segments = extract_pred_wall_segments(node_info, scale)
    pred_name_to_index = {name: i for i, (name, _) in enumerate(pred_planes)}
    gt_name_to_index = {}
    for i, (name, _) in enumerate(gt_planes):
        parts = name.split("_")
        if len(parts) >= 2:
            gt_name_to_index[f"gt_{parts[1]}"] = i

    points = [p for _, p0, p1 in gt_segments + pred_segments for p in (p0, p1)]
    if points:
        pts = np.asarray(points)
        center = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - center, axis=1)
        keep = dist < max(1.0, np.percentile(dist, 95) * 1.5)
        pts = pts[keep] if keep.any() else pts
        xmin, zmin = pts.min(axis=0)
        xmax, zmax = pts.max(axis=0)
    else:
        xmin, xmax, zmin, zmax = -1, 1, -1, 1
    pad = max(xmax - xmin, zmax - zmin, 0.5) * 0.18
    xmin, xmax, zmin, zmax = xmin - pad, xmax + pad, zmin - pad, zmax + pad

    fig, ax = plt.subplots(figsize=(7, 7))
    for name, p0, p1 in gt_segments:
        gi = gt_name_to_index.get(name)
        color = "#2ca02c" if gi in matched_gt else "#9e9e9e"
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=7, alpha=0.55)
        mid = (p0 + p1) / 2
        ax.text(mid[0], mid[1], name, color=color, fontsize=8)

    for name, p0, p1 in pred_segments:
        pi = pred_name_to_index.get(name)
        color = "#1f77b4" if pi in matched_pred else "#d62728"
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=3, linestyle="--")
        mid = (p0 + p1) / 2
        ax.text(mid[0], mid[1], name, color=color, fontsize=8)

    pred_count = len(pred_planes)
    gt_count = len(gt_planes)
    precision = len(matched_pred) / pred_count if pred_count else 0
    recall = len(matched_gt) / gt_count if gt_count else 0
    ax.set_title(
        f"{result_dir.parent.name}/{result_dir.name} annotation floor plan\n"
        f"plane matched {len(matched_pred)} | pred {pred_count} | gt {gt_count} | "
        f"P {precision:.3f} / R {recall:.3f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.plot([], [], color="#2ca02c", lw=7, alpha=0.55, label="Annotation GT matched")
    ax.plot([], [], color="#9e9e9e", lw=7, alpha=0.55, label="Annotation GT unmatched")
    ax.plot([], [], color="#1f77b4", lw=3, linestyle="--", label="Pred matched")
    ax.plot([], [], color="#d62728", lw=3, linestyle="--", label="Pred unmatched")
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path = result_dir / output_name
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_floorplan_segments(
    result_dir,
    gt_room_path,
    node_info,
    rt0,
    scale,
    pred_planes,
    gt_planes,
    matched_pred,
    matched_gt,
    output_name,
):
    gt_segments = extract_gt_wall_segments(gt_room_path, rt0)
    pred_segments = extract_pred_wall_segments(node_info, scale)
    pred_name_to_index = {name: i for i, (name, _) in enumerate(pred_planes)}
    gt_name_to_index = {name: i for i, (name, _) in enumerate(gt_planes)}

    points = [p for _, p0, p1 in gt_segments + pred_segments for p in (p0, p1)]
    if points:
        pts = np.asarray(points)
        xmin, zmin = pts.min(axis=0)
        xmax, zmax = pts.max(axis=0)
    else:
        xmin, xmax, zmin, zmax = -1, 1, -1, 1
    pad = max(xmax - xmin, zmax - zmin, 0.5) * 0.18
    xmin, xmax, zmin, zmax = xmin - pad, xmax + pad, zmin - pad, zmax + pad

    fig, ax = plt.subplots(figsize=(7, 7))
    for name, p0, p1 in gt_segments:
        gi = gt_name_to_index.get(name)
        color = "#2ca02c" if gi in matched_gt else "#9e9e9e"
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=7, alpha=0.55)
        mid = (p0 + p1) / 2
        ax.text(mid[0], mid[1], name, color=color, fontsize=8)

    for name, p0, p1 in pred_segments:
        pi = pred_name_to_index.get(name)
        color = "#1f77b4" if pi in matched_pred else "#d62728"
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=3, linestyle="--")
        mid = (p0 + p1) / 2
        ax.text(mid[0], mid[1], name, color=color, fontsize=8)

    pred_count = len(pred_planes)
    gt_count = len(gt_planes)
    precision = len(matched_pred) / pred_count if pred_count else 0
    recall = len(matched_gt) / gt_count if gt_count else 0
    ax.set_title(
        f"{result_dir.parent.name}/{result_dir.name} floor-plan view\n"
        f"plane matched {len(matched_pred)} | pred {pred_count} | gt {gt_count} | "
        f"P {precision:.3f} / R {recall:.3f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.plot([], [], color="#2ca02c", lw=7, alpha=0.55, label="GT matched wall segment")
    ax.plot([], [], color="#9e9e9e", lw=7, alpha=0.55, label="GT unmatched wall segment")
    ax.plot([], [], color="#1f77b4", lw=3, linestyle="--", label="Pred matched wall segment")
    ax.plot([], [], color="#d62728", lw=3, linestyle="--", label="Pred unmatched wall segment")
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path = result_dir / output_name
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_room(
    result_dir,
    gt_room_path,
    output_name,
    floorplan_output_name,
    annotation_floorplan_output_name,
    annotation,
):
    node_path = result_dir / "node_data.json"
    dust3r_path = result_dir / "dust3r_output.npz"
    if not node_path.exists() or not dust3r_path.exists():
        return None

    node_info = json.loads(node_path.read_text())
    dust3r_output = np.load(dust3r_path)
    poses = dust3r_output["poses"]
    ref_idx = closest_pose_index(poses)
    scale, rt0 = estimate_metric_scale(poses, gt_room_path, ref_idx)

    gt_dict = extract_scene_planes(gt_room_path, rt0)
    gt_planes = [(f"gt_{pid}", np.asarray(plane, dtype=np.float64)) for pid, plane in gt_dict.items()]
    pred_planes = collect_pred_planes(node_info, scale)
    matched_pred, matched_gt, pairs = match_planes(pred_planes, gt_planes)

    gt_walls = [(i, name, plane_to_xz_line(plane)) for i, (name, plane) in enumerate(gt_planes) if is_wall(plane)]
    pred_walls = [
        (i, name, plane_to_xz_line(plane)) for i, (name, plane) in enumerate(pred_planes) if is_wall(plane)
    ]

    points = []
    lines = [line for _, _, line in gt_walls + pred_walls]
    for info in node_info.get("global_plane_info", []):
        for key in ("left_endpoint", "right_endpoint"):
            if key in info and info[key]:
                endpoint = np.asarray(info[key], dtype=np.float64)
                points.append(endpoint[[0, 2]] * scale)
    if len(points) < 2:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                point = intersect_lines(lines[i], lines[j])
                if point is not None and np.all(np.isfinite(point)):
                    points.append(point)

    if points:
        pts = np.asarray(points)
        center = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - center, axis=1)
        pts = pts[dist < max(1.0, np.percentile(dist, 90) * 1.5)]
        xmin, zmin = pts.min(axis=0)
        xmax, zmax = pts.max(axis=0)
    else:
        xmin, xmax, zmin, zmax = -1, 1, -1, 1
    pad = max(xmax - xmin, zmax - zmin, 0.5) * 0.2
    xmin, xmax, zmin, zmax = xmin - pad, xmax + pad, zmin - pad, zmax + pad

    fig, ax = plt.subplots(figsize=(7, 7))
    for gi, name, line in gt_walls:
        seg = line_segment_in_bbox(line, xmin, xmax, zmin, zmax)
        if not seg:
            continue
        color = "#2ca02c" if gi in matched_gt else "#9e9e9e"
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color=color, lw=5, alpha=0.65)
        mid = ((seg[0][0] + seg[1][0]) / 2, (seg[0][1] + seg[1][1]) / 2)
        ax.text(*mid, name, color=color, fontsize=8)

    for pi, name, line in pred_walls:
        seg = line_segment_in_bbox(line, xmin, xmax, zmin, zmax)
        if not seg:
            continue
        color = "#1f77b4" if pi in matched_pred else "#d62728"
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color=color, lw=2.5, linestyle="--")
        mid = ((seg[0][0] + seg[1][0]) / 2, (seg[0][1] + seg[1][1]) / 2)
        ax.text(mid[0], mid[1], name, color=color, fontsize=8)

    pred_count = len(pred_planes)
    gt_count = len(gt_planes)
    precision = len(matched_pred) / pred_count if pred_count else 0
    recall = len(matched_gt) / gt_count if gt_count else 0
    title = (
        f"{result_dir.parent.name}/{result_dir.name} plane merge\n"
        f"matched {len(pairs)} | pred {pred_count} | gt {gt_count} | "
        f"P {precision:.3f} / R {recall:.3f}"
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.plot([], [], color="#2ca02c", lw=5, alpha=0.65, label="GT matched")
    ax.plot([], [], color="#9e9e9e", lw=5, alpha=0.65, label="GT unmatched")
    ax.plot([], [], color="#1f77b4", lw=2.5, linestyle="--", label="Pred matched")
    ax.plot([], [], color="#d62728", lw=2.5, linestyle="--", label="Pred unmatched")
    ax.legend(loc="upper right")

    output_path = result_dir / output_name
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    floorplan_output = plot_floorplan_segments(
        result_dir,
        gt_room_path,
        node_info,
        rt0,
        scale,
        pred_planes,
        gt_planes,
        matched_pred,
        matched_gt,
        floorplan_output_name,
    )
    outputs = [output_path, floorplan_output]
    if annotation:
        annotation_output = plot_annotation_floorplan(
            result_dir,
            annotation,
            node_info,
            rt0,
            scale,
            pred_planes,
            annotation_floorplan_output_name,
        )
        outputs.append(annotation_output)
    return outputs


def main():
    args = parse_args()
    result_root = Path(args.result_root)
    gt_root = Path(args.gt_root)
    outputs = []

    for scene_dir in sorted(p for p in result_root.iterdir() if p.is_dir()):
        scene_name = scene_dir.name
        if not scene_name.startswith(args.scene_prefix):
            scene_name = f"{args.scene_prefix}{scene_name}"
        annotation = load_annotation(gt_root / scene_name / "annotation_3d.json")
        for result_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
            gt_room_path = gt_root / scene_name / "2D_rendering" / result_dir.name / "perspective" / "full"
            if not gt_room_path.exists():
                print(f"skip missing GT: {gt_room_path}")
                continue
            output = plot_room(
                result_dir,
                gt_room_path,
                args.output_name,
                args.floorplan_output_name,
                args.annotation_floorplan_output_name,
                annotation,
            )
            if output:
                outputs.extend(output)
                for item in output:
                    print(item)

    print(f"wrote {len(outputs)} visualization(s)")


if __name__ == "__main__":
    main()
