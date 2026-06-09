# Learnable Plane Merge

This branch adds a first learnable merge baseline on top of the original Plane-DUSt3R pipeline.

The module does not change the DUSt3R backbone or the single-view NonCuboid detector. It learns to score and optionally refine global wall candidates produced by the geometric merge stage.

## Pipeline

```text
RGB views
  -> DUSt3R geometry
  -> single-view plane detection
  -> geometric candidate merge
  -> LearnablePlaneMerge wall scorer/refiner
  -> 2D floor-plan evaluation
```

## 1. Build candidate results

Use an existing cached result root, or generate conservative candidates:

```bash
python recompute_plane_merge.py \
  --source_result_root eval_results_sample500 \
  --output_root eval_results_sample500_conservative \
  --merge_variant conservative \
  --force
```

## 2. Export supervised tokens

```bash
python export_learnable_merge_data.py \
  --result_root eval_results_sample500_conservative \
  --gt_root ~/datasets/Structured3D \
  --output_dir learnable_merge_data_sample500
```

Each `.npz` sample contains wall candidate features, keep/drop labels, and matched GT plane/endpoint targets from `annotation_3d.json`.

## 3. Train

```bash
python train_learnable_plane_merge.py \
  --data_dir learnable_merge_data_sample500 \
  --output checkpoints/learnable_plane_merge_sample500.pt \
  --epochs 50 \
  --batch_size 16 \
  --device cuda
```

## 4. Apply

```bash
python apply_learnable_plane_merge.py \
  --source_result_root eval_results_sample500_conservative \
  --gt_root ~/datasets/Structured3D \
  --output_root eval_results_sample500_learnable \
  --checkpoint checkpoints/learnable_plane_merge_sample500.pt \
  --keep_threshold 0.5 \
  --device cuda
```

Use `--refine` to also apply predicted plane and endpoint deltas. For the first ablation, compare filtering-only and refine variants separately.

If plain score thresholding improves precision but hurts recall, use the hybrid weak-candidate mode:

```bash
python apply_learnable_plane_merge.py \
  --source_result_root eval_results_sample500_conservative \
  --gt_root ~/datasets/Structured3D \
  --output_root eval_results_sample500_learnable_weak \
  --checkpoint checkpoints/learnable_plane_merge_sample500.pt \
  --filter_mode weak_score \
  --keep_threshold 0.3 \
  --weak_short_ratio 0.5 \
  --device cuda
```

This keeps strongly supported wall candidates by default and only asks the model to remove short, single-view weak candidates.

## 5. Evaluate

```bash
python evaluate_floorplan.py \
  --result_root eval_results_sample500_learnable \
  --gt_root ~/datasets/Structured3D \
  --output_csv floorplan_metrics_sample500_learnable.csv

python compare_floorplan_metrics.py \
  --baseline_csv floorplan_metrics_sample500.csv \
  --candidate_csv floorplan_metrics_sample500_learnable.csv \
  --output_csv floorplan_metric_delta_learnable.csv
```

Primary metrics to watch:

```text
wall_precision
wall_recall
floorplan_polygon_iou
mean_endpoint_error_m
pred_walls
matched_walls
```
