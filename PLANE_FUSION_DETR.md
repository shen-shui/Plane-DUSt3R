# Plane Fusion DETR

This branch starts a stronger network direction for RGB multi-view to 2D floor-plan estimation.

Instead of only filtering rule-based merged walls, `PlaneFusionDETR` predicts a room-level set of global 2D wall lines with learned queries.

```text
multi-view plane candidate tokens
  -> Transformer encoder
  -> learned wall queries / Transformer decoder
  -> wall objectness + 2D line + endpoints
  -> node_data.json for existing floor-plan evaluation
```

The first version trains from the `.npz` files exported by `export_learnable_merge_data.py`.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python train_plane_fusion.py \
  --data_dir learnable_merge_data_sample2000 \
  --output checkpoints/plane_fusion_detr_sample2000.pt \
  --epochs 100 \
  --batch_size 16 \
  --line_weight 2.0 \
  --endpoint_weight 0.5 \
  --consistency_weight 0.5 \
  --device cuda
```

## Apply To Sample500

```bash
CUDA_VISIBLE_DEVICES=0 python apply_plane_fusion.py \
  --source_result_root eval_results_sample500_conservative \
  --data_dir learnable_merge_data_sample500 \
  --output_root eval_results_sample500_plane_fusion \
  --checkpoint checkpoints/plane_fusion_detr_sample2000.pt \
  --score_threshold 0.5 \
  --device cuda
```

If too few or too many walls are produced, sweep `--score_threshold`:

```bash
for t in 0.2 0.3 0.4 0.5 0.6; do
  tag=${t/./}
  CUDA_VISIBLE_DEVICES=0 python apply_plane_fusion.py \
    --source_result_root eval_results_sample500_conservative \
    --data_dir learnable_merge_data_sample500 \
    --output_root eval_results_sample500_plane_fusion_t${tag} \
    --checkpoint checkpoints/plane_fusion_detr_sample2000.pt \
    --score_threshold $t \
    --device cuda

  python evaluate_floorplan.py \
    --result_root eval_results_sample500_plane_fusion_t${tag} \
    --gt_root ~/datasets/Structured3D \
    --output_csv floorplan_metrics_sample500_plane_fusion_t${tag}.csv

  python compare_floorplan_metrics.py \
    --baseline_csv floorplan_metrics_sample500.csv \
    --candidate_csv floorplan_metrics_sample500_plane_fusion_t${tag}.csv \
    --output_csv floorplan_metric_delta_plane_fusion_t${tag}.csv
done
```

## Notes

This is intentionally separate from DUSt3R backbone training. It targets the failure mode observed in evaluation: multi-view wall merge and 2D floor-plan formation.

Use both metrics when comparing outputs:

```text
floorplan_polygon_iou: area overlap of the inferred polygon; can be optimistic for messy wall sets.
floorplan_line_iou: rasterized wall-line overlap; stricter for actual wall alignment.
```

If this query-based wall head improves `floorplan_polygon_iou`, the next step is to connect it more tightly to DUSt3R/plane detector features instead of using cached candidate tokens.
