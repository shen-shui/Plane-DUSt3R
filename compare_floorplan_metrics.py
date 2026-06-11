import argparse
import csv
from pathlib import Path

import numpy as np


METRIC_KEYS = [
    "wall_precision",
    "wall_recall",
    "floorplan_polygon_iou",
    "mean_angle_error_deg",
    "mean_offset_error_m",
    "mean_endpoint_error_m",
    "pred_walls",
    "matched_walls",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two floor-plan metric CSV files.")
    parser.add_argument("--baseline_csv", required=True)
    parser.add_argument("--candidate_csv", required=True)
    parser.add_argument("--output_csv", default="floorplan_metric_delta.csv")
    return parser.parse_args()


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_rows(path):
    rows = {}
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            rows[(row["scene"], row["room"])] = row
    return rows


def summarize(rows, label, view_filter=None):
    selected = []
    for row in rows:
        views = parse_float(row.get("views"))
        if view_filter is not None and not view_filter(views):
            continue
        selected.append(row)
    print(f"{label}: n={len(selected)}")
    for key in METRIC_KEYS:
        base_values = np.asarray([parse_float(row.get(f"baseline_{key}")) for row in selected], dtype=np.float64)
        cand_values = np.asarray([parse_float(row.get(f"candidate_{key}")) for row in selected], dtype=np.float64)
        delta_values = np.asarray([parse_float(row.get(f"delta_{key}")) for row in selected], dtype=np.float64)
        finite = np.isfinite(base_values) & np.isfinite(cand_values) & np.isfinite(delta_values)
        if finite.any():
            print(
                f"  {key}: "
                f"baseline={base_values[finite].mean():.4f}, "
                f"candidate={cand_values[finite].mean():.4f}, "
                f"delta={delta_values[finite].mean():+.4f}"
            )


def main():
    args = parse_args()
    baseline = load_rows(Path(args.baseline_csv))
    candidate = load_rows(Path(args.candidate_csv))
    common_keys = sorted(set(baseline) & set(candidate))

    rows = []
    for key in common_keys:
        base = baseline[key]
        cand = candidate[key]
        row = {
            "scene": key[0],
            "room": key[1],
            "views": cand.get("views", base.get("views", "")),
        }
        for metric in METRIC_KEYS:
            base_value = parse_float(base.get(metric))
            cand_value = parse_float(cand.get(metric))
            row[f"baseline_{metric}"] = base_value
            row[f"candidate_{metric}"] = cand_value
            row[f"delta_{metric}"] = cand_value - base_value
        rows.append(row)

    fields = ["scene", "room", "views"]
    for metric in METRIC_KEYS:
        fields.extend([f"baseline_{metric}", f"candidate_{metric}", f"delta_{metric}"])

    with open(args.output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"matched rooms: {len(rows)}")
    print(f"wrote {args.output_csv}")
    summarize(rows, "all")
    summarize(rows, "views >= 3", lambda views: views >= 3)
    summarize(rows, "views >= 5", lambda views: views >= 5)


if __name__ == "__main__":
    main()
