import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from plane_merge_planedust3r import plane_merge


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute Plane-DUSt3R plane merge from cached dust3r and plane-detection outputs."
    )
    parser.add_argument("--source_result_root", required=True, help="Existing evaluation result root.")
    parser.add_argument("--output_root", required=True, help="Output root for recomputed node_data.json files.")
    parser.add_argument(
        "--merge_variant",
        choices=["default", "conservative"],
        default="conservative",
        help="Plane merge strategy to run.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output node_data.json.")
    parser.add_argument("--no_links", action="store_true", help="Copy cache files instead of creating symlinks.")
    parser.add_argument("--metric", action="store_true", help="Use metric-scale thresholds in plane_merge.")
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


def iter_room_dirs(root):
    for scene_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for room_dir in sorted(path for path in scene_dir.iterdir() if path.is_dir()):
            yield scene_dir, room_dir


def main():
    args = parse_args()
    source_root = Path(args.source_result_root)
    output_root = Path(args.output_root)

    room_dirs = list(iter_room_dirs(source_root))
    recomputed = 0
    skipped = 0
    failed = 0

    for scene_dir, room_dir in tqdm(room_dirs):
        rel_dir = room_dir.relative_to(source_root)
        out_dir = output_root / rel_dir
        out_node_path = out_dir / "node_data.json"
        dust3r_path = room_dir / "dust3r_output.npz"
        plane_detection_path = room_dir / "plane_detection.json"

        if out_node_path.exists() and not args.force:
            skipped += 1
            continue
        if not dust3r_path.exists() or not plane_detection_path.exists():
            failed += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        link_or_copy(dust3r_path, out_dir / "dust3r_output.npz", args.no_links)
        link_or_copy(plane_detection_path, out_dir / "plane_detection.json", args.no_links)
        if (room_dir / "metric_results.txt").exists():
            link_or_copy(room_dir / "metric_results.txt", out_dir / "metric_results.txt", args.no_links)

        try:
            dust3r_output = np.load(dust3r_path)
            plane_detection = json.loads(plane_detection_path.read_text())
            plane_merge(
                dust3r_output,
                plane_detection,
                save=True,
                filedir=str(out_dir),
                metric=args.metric,
                merge_variant=args.merge_variant,
            )
            recomputed += 1
        except Exception as exc:
            failed += 1
            print(f"failed {rel_dir}: {exc}")

    print(
        f"done: recomputed={recomputed}, skipped={skipped}, failed={failed}, "
        f"output={output_root}"
    )


if __name__ == "__main__":
    main()
