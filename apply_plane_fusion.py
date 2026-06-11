import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from plane_fusion.model import PlaneFusionDETR


def parse_args():
    parser = argparse.ArgumentParser(description="Apply PlaneFusionDETR and write Plane-DUSt3R node_data.json files.")
    parser.add_argument("--source_result_root", required=True)
    parser.add_argument("--data_dir", required=True, help="Exported plane fusion .npz token directory.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--min_walls", type=int, default=3)
    parser.add_argument("--max_walls", type=int, default=12)
    parser.add_argument("--scene_prefix", default="scene_")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_links", action="store_true")
    return parser.parse_args()


def link_or_copy(src, dst, no_links=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if no_links:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["args"]
    model = PlaneFusionDETR(
        input_dim=checkpoint["input_dim"],
        hidden_dim=cfg.get("hidden_dim", 256),
        num_encoder_layers=cfg.get("num_encoder_layers", 3),
        num_decoder_layers=cfg.get("num_decoder_layers", 3),
        num_heads=cfg.get("num_heads", 8),
        num_queries=cfg.get("num_queries", 16),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def iter_room_dirs(root):
    for scene_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for room_dir in sorted(path for path in scene_dir.iterdir() if path.is_dir()):
            yield scene_dir, room_dir


def infer_scale(node_info, features):
    walls = node_info.get("global_plane_info", [])
    for wall, feature in zip(walls, features):
        left = wall.get("left_endpoint")
        right = wall.get("right_endpoint")
        if not left or not right:
            continue
        unscaled = np.linalg.norm(np.asarray(right, dtype=np.float64)[[0, 2]] - np.asarray(left, dtype=np.float64)[[0, 2]])
        metric = np.linalg.norm(np.asarray(feature[6:8], dtype=np.float64) - np.asarray(feature[4:6], dtype=np.float64))
        if unscaled > 1e-6 and metric > 1e-6:
            return float(metric / unscaled)
    return 1.0


def wall_y_value(node_info):
    values = []
    for wall in node_info.get("global_plane_info", []):
        for key in ("left_endpoint", "right_endpoint"):
            point = wall.get(key)
            if point and len(point) >= 3:
                values.append(float(point[1]))
    return float(np.median(values)) if values else 0.0


def line_from_endpoints(endpoints):
    endpoints = np.asarray(endpoints, dtype=np.float64)
    p0 = endpoints[:2]
    p1 = endpoints[2:]
    direction = p1 - p0
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return None
    normal = np.array([direction[1], -direction[0]], dtype=np.float64) / length
    center = (p0 + p1) / 2
    offset = -float(np.dot(normal, center))
    return np.array([normal[0], normal[1], offset], dtype=np.float64)


def make_wall(index, line, endpoints, scale, y_value, score):
    line = np.asarray(line, dtype=np.float64)
    endpoints = np.asarray(endpoints, dtype=np.float64)
    p0 = endpoints[:2] / scale
    p1 = endpoints[2:] / scale
    endpoint_line = line_from_endpoints(endpoints)
    line_norm = np.linalg.norm(line[:2])
    if line_norm > 1e-6:
        line = line / line_norm
    if endpoint_line is not None:
        if line_norm > 1e-6 and np.dot(endpoint_line[:2], line[:2]) < 0:
            endpoint_line = -endpoint_line
        line = endpoint_line
    else:
        line = line if line_norm > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return {
        "index": index,
        "pparam": [float(line[0]), 0.0, float(line[1]), float(line[2] / scale)],
        "pre": None,
        "next": None,
        "left_endpoint": [float(p0[0]), y_value, float(p0[1])],
        "right_endpoint": [float(p1[0]), y_value, float(p1[1])],
        "plane_fusion_score": float(score),
    }


def apply_room(model, room_dir, data_path, out_dir, args):
    node_path = room_dir / "node_data.json"
    if not node_path.exists() or not data_path.exists():
        return False
    node_info = json.loads(node_path.read_text())
    data = np.load(data_path, allow_pickle=True)
    features = data["features"].astype(np.float32)
    if len(features) == 0:
        return False

    with torch.no_grad():
        feature_tensor = torch.from_numpy(features).unsqueeze(0).to(args.device)
        mask = torch.ones(1, feature_tensor.shape[1], dtype=torch.bool, device=args.device)
        outputs = model(feature_tensor, mask)
        scores = torch.sigmoid(outputs["logits"])[0].cpu().numpy()
        lines = outputs["line"][0].cpu().numpy()
        endpoints = outputs["endpoints"][0].cpu().numpy()

    keep = np.where(scores >= args.score_threshold)[0].tolist()
    if len(keep) < args.min_walls:
        keep = np.argsort(scores)[-args.min_walls:].tolist()
    keep = sorted(keep, key=lambda idx: scores[idx], reverse=True)[: args.max_walls]

    scale = infer_scale(node_info, features)
    y_value = wall_y_value(node_info)
    walls = [
        make_wall(new_idx, lines[idx], endpoints[idx], scale, y_value, scores[idx])
        for new_idx, idx in enumerate(keep)
    ]
    node_info["global_plane_info"] = walls
    node_info["planes"] = {str(key): [] for key in node_info.get("planes", {})}
    node_info["plane_fusion"] = {
        "checkpoint": str(args.checkpoint),
        "score_threshold": args.score_threshold,
        "min_walls": args.min_walls,
        "max_walls": args.max_walls,
        "input_candidates": int(len(features)),
        "output_walls": int(len(walls)),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "node_data.json", "w") as file:
        json.dump(node_info, file, indent=4)
    return True


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    model = load_model(args.checkpoint, args.device)
    source_root = Path(args.source_result_root)
    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)

    applied = 0
    failed = 0
    for scene_dir, room_dir in tqdm(list(iter_room_dirs(source_root))):
        scene_name = scene_dir.name if scene_dir.name.startswith(args.scene_prefix) else f"{args.scene_prefix}{scene_dir.name}"
        data_path = data_dir / f"{scene_name}_{room_dir.name}.npz"
        out_dir = output_root / room_dir.relative_to(source_root)
        for filename in ("dust3r_output.npz", "plane_detection.json", "metric_results.txt"):
            src = room_dir / filename
            if src.exists():
                link_or_copy(src, out_dir / filename, args.no_links)
        try:
            if apply_room(model, room_dir, data_path, out_dir, args):
                applied += 1
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            print(f"failed {scene_name}/{room_dir.name}: {exc}")
    print(f"applied={applied}, failed={failed}, output={output_root}")


if __name__ == "__main__":
    main()
