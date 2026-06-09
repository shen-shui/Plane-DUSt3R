import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from learnable_plane_merge.features import FEATURE_DIM, wall_info_to_base_targets, wall_info_to_feature
from learnable_plane_merge.model import LearnablePlaneMerge
from visualize_plane_merge import closest_pose_index, estimate_metric_scale


def parse_args():
    parser = argparse.ArgumentParser(description="Apply a trained learnable plane merge scorer.")
    parser.add_argument("--source_result_root", required=True)
    parser.add_argument("--gt_root", required=True, help="Structured3D root, used only to recover metric scale.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--keep_threshold", type=float, default=0.5)
    parser.add_argument("--min_walls", type=int, default=3)
    parser.add_argument("--refine", action="store_true", help="Apply predicted plane/endpoint deltas.")
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
    args = checkpoint.get("args", {})
    model = LearnablePlaneMerge(
        input_dim=checkpoint.get("input_dim", FEATURE_DIM),
        hidden_dim=args.get("hidden_dim", 128),
        num_layers=args.get("num_layers", 3),
        num_heads=args.get("num_heads", 4),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def iter_room_dirs(root):
    for scene_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for room_dir in sorted(path for path in scene_dir.iterdir() if path.is_dir()):
            yield scene_dir, room_dir


def estimate_room_scale(room_dir, gt_root, scene_name):
    dust3r_path = room_dir / "dust3r_output.npz"
    gt_room_path = gt_root / scene_name / "2D_rendering" / room_dir.name / "perspective" / "full"
    if not dust3r_path.exists() or not gt_room_path.exists():
        return 1.0
    dust3r_output = np.load(dust3r_path)
    poses = dust3r_output["poses"]
    ref_idx = closest_pose_index(poses)
    scale, _ = estimate_metric_scale(poses, gt_room_path, ref_idx)
    return scale


def apply_room(model, room_dir, out_dir, gt_root, scene_name, args):
    node_path = room_dir / "node_data.json"
    if not node_path.exists():
        return False
    node_info = json.loads(node_path.read_text())
    walls = node_info.get("global_plane_info", [])
    if not walls:
        return False

    scale = estimate_room_scale(room_dir, gt_root, scene_name)
    features = []
    base_planes = []
    base_endpoints = []
    valid_indices = []
    for idx, wall in enumerate(walls):
        if not wall.get("left_endpoint") or not wall.get("right_endpoint"):
            continue
        features.append(wall_info_to_feature(wall, scale))
        base_p, base_e = wall_info_to_base_targets(wall, scale)
        base_planes.append(base_p)
        base_endpoints.append(base_e)
        valid_indices.append(idx)
    if not features:
        return False

    with torch.no_grad():
        feature_tensor = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0).to(args.device)
        mask = torch.ones(1, feature_tensor.shape[1], dtype=torch.bool, device=args.device)
        outputs = model(feature_tensor, mask)
        scores = torch.sigmoid(outputs["keep_logits"])[0].cpu().numpy()
        plane_delta = outputs["plane_delta"][0].cpu().numpy()
        endpoint_delta = outputs["endpoint_delta"][0].cpu().numpy()

    keep_local = scores >= args.keep_threshold
    if keep_local.sum() < args.min_walls:
        keep_local[np.argsort(scores)[-args.min_walls:]] = True

    old_to_new = {}
    new_walls = []
    for local_idx, old_idx in enumerate(valid_indices):
        if not keep_local[local_idx]:
            continue
        wall = dict(walls[old_idx])
        old_to_new[old_idx] = len(new_walls)
        wall["learned_keep_score"] = float(scores[local_idx])
        if args.refine:
            plane = np.asarray(base_planes[local_idx]) + plane_delta[local_idx]
            endpoints = np.asarray(base_endpoints[local_idx]) + endpoint_delta[local_idx]
            plane_unscaled = plane.copy()
            plane_unscaled[3] /= scale
            wall["pparam"] = plane_unscaled.tolist()
            left = np.asarray(wall["left_endpoint"], dtype=np.float64).copy()
            right = np.asarray(wall["right_endpoint"], dtype=np.float64).copy()
            left[[0, 2]] = endpoints[:2] / scale
            right[[0, 2]] = endpoints[2:] / scale
            wall["left_endpoint"] = left.tolist()
            wall["right_endpoint"] = right.tolist()
        new_walls.append(wall)

    for new_idx, wall in enumerate(new_walls):
        old_pre = wall.get("pre")
        old_next = wall.get("next")
        wall["index"] = new_idx
        wall["pre"] = old_to_new.get(old_pre)
        wall["next"] = old_to_new.get(old_next)

    node_info["global_plane_info"] = new_walls
    node_info["learnable_plane_merge"] = {
        "checkpoint": str(args.checkpoint),
        "keep_threshold": args.keep_threshold,
        "refine": args.refine,
        "input_walls": len(walls),
        "output_walls": len(new_walls),
    }
    node_info["planes"] = {
        str(img_id): [old_to_new[idx] for idx in ids if idx in old_to_new]
        for img_id, ids in node_info.get("planes", {}).items()
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
    output_root = Path(args.output_root)
    gt_root = Path(args.gt_root)

    applied = 0
    failed = 0
    for scene_dir, room_dir in tqdm(list(iter_room_dirs(source_root))):
        scene_name = scene_dir.name
        if not scene_name.startswith(args.scene_prefix):
            scene_name = f"{args.scene_prefix}{scene_name}"
        out_dir = output_root / room_dir.relative_to(source_root)
        for filename in ("dust3r_output.npz", "plane_detection.json", "metric_results.txt"):
            src = room_dir / filename
            if src.exists():
                link_or_copy(src, out_dir / filename, args.no_links)
        try:
            if apply_room(model, room_dir, out_dir, gt_root, scene_name, args):
                applied += 1
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            print(f"failed {scene_name}/{room_dir.name}: {exc}")
    print(f"applied={applied}, failed={failed}, output={output_root}")


if __name__ == "__main__":
    main()
