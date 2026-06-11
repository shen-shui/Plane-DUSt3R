import torch
import torch.nn.functional as F


def line_angle_distance(pred, target):
    dot = (pred[..., None, :2] * target[..., None, :, :2]).sum(dim=-1).abs().clamp(max=1.0)
    return 1.0 - dot


def endpoint_l1_cost(pred, target):
    direct = torch.cdist(pred, target, p=1)
    flipped_target = torch.cat([target[..., 2:], target[..., :2]], dim=-1)
    flipped = torch.cdist(pred, flipped_target, p=1)
    return torch.minimum(direct, flipped)


def sample_segment_points(endpoints, num_samples=16):
    p0 = endpoints[..., :2]
    p1 = endpoints[..., 2:]
    steps = torch.linspace(0.0, 1.0, num_samples, device=endpoints.device, dtype=endpoints.dtype)
    view_shape = (1,) * p0.dim() + (num_samples,)
    steps = steps.view(*view_shape).movedim(-1, -2)
    return p0.unsqueeze(-2) * (1.0 - steps) + p1.unsqueeze(-2) * steps


def segment_chamfer_distance(pred, target, num_samples=16):
    pred_points = sample_segment_points(pred, num_samples=num_samples)
    target_points = sample_segment_points(target, num_samples=num_samples)
    distances = torch.cdist(pred_points, target_points, p=2)
    return 0.5 * (distances.min(dim=-1).values.mean(dim=-1) + distances.min(dim=-2).values.mean(dim=-1))


def line_from_endpoints(endpoints):
    p0 = endpoints[..., :2]
    p1 = endpoints[..., 2:]
    direction = p1 - p0
    length = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    normal = torch.stack([direction[..., 1], -direction[..., 0]], dim=-1) / length
    center = (p0 + p1) / 2
    offset = -(normal * center).sum(dim=-1, keepdim=True)
    return torch.cat([normal, offset], dim=-1)


def oriented_line_l1(pred, target):
    direct = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
    flipped = F.smooth_l1_loss(pred, -target, reduction="none").mean(dim=-1)
    return torch.minimum(direct, flipped)


def match_single(cost):
    cost_np = cost.detach().cpu().numpy()
    try:
        from scipy.optimize import linear_sum_assignment

        row, col = linear_sum_assignment(cost_np)
        return list(zip(row.tolist(), col.tolist()))
    except Exception:
        pairs = []
        used_rows = set()
        used_cols = set()
        flat = cost.flatten().argsort()
        n_rows, n_cols = cost.shape
        for item in flat.tolist():
            row = item // n_cols
            col = item % n_cols
            if row in used_rows or col in used_cols:
                continue
            pairs.append((row, col))
            used_rows.add(row)
            used_cols.add(col)
            if len(used_cols) == min(n_rows, n_cols):
                break
        return pairs


def plane_fusion_loss(
    outputs,
    batch,
    line_weight=2.0,
    endpoint_weight=0.5,
    consistency_weight=0.5,
    chamfer_weight=1.0,
    chamfer_samples=16,
    no_object_weight=0.1,
):
    logits = outputs["logits"]
    pred_lines = outputs["line"]
    pred_endpoints = outputs["endpoints"]
    target_lines = batch["target_lines"]
    target_endpoints = batch["target_endpoints"]
    target_mask = batch["target_mask"]

    cls_targets = torch.zeros_like(logits)
    line_losses = []
    endpoint_losses = []
    consistency_losses = []
    chamfer_losses = []

    for b in range(logits.shape[0]):
        valid = target_mask[b]
        if not valid.any():
            continue
        tgt_lines = target_lines[b, valid]
        tgt_endpoints = target_endpoints[b, valid]
        cost = (
            line_angle_distance(pred_lines[b], tgt_lines)
            + torch.cdist(pred_lines[b, :, 2:3], tgt_lines[:, 2:3], p=1)
            + 0.25 * endpoint_l1_cost(pred_endpoints[b], tgt_endpoints)
            + 0.25 * segment_chamfer_distance(
                pred_endpoints[b, :, None, :],
                tgt_endpoints[None, :, :],
                num_samples=chamfer_samples,
            )
            - logits[b].sigmoid().unsqueeze(-1)
        )
        if cost.shape != (pred_lines.shape[1], tgt_lines.shape[0]):
            raise RuntimeError(f"Unexpected matching cost shape: {tuple(cost.shape)}")
        pairs = match_single(cost)
        for pred_idx, tgt_idx in pairs:
            cls_targets[b, pred_idx] = 1.0
            line_losses.append(F.smooth_l1_loss(pred_lines[b, pred_idx], tgt_lines[tgt_idx], reduction="none").mean())
            direct = F.smooth_l1_loss(
                pred_endpoints[b, pred_idx], tgt_endpoints[tgt_idx], reduction="none"
            ).mean()
            flipped_tgt = torch.cat([tgt_endpoints[tgt_idx, 2:], tgt_endpoints[tgt_idx, :2]], dim=0)
            flipped = F.smooth_l1_loss(pred_endpoints[b, pred_idx], flipped_tgt, reduction="none").mean()
            endpoint_losses.append(torch.minimum(direct, flipped))
            chamfer_losses.append(
                segment_chamfer_distance(
                    pred_endpoints[b, pred_idx],
                    tgt_endpoints[tgt_idx],
                    num_samples=chamfer_samples,
                )
            )
            endpoint_line = line_from_endpoints(pred_endpoints[b, pred_idx])
            consistency_losses.append(oriented_line_l1(pred_lines[b, pred_idx], endpoint_line))

    pos_weight = torch.ones_like(logits)
    pos_weight[cls_targets < 0.5] = no_object_weight
    cls_loss = F.binary_cross_entropy_with_logits(logits, cls_targets, weight=pos_weight)
    if line_losses:
        line_loss = torch.stack(line_losses).mean()
        endpoint_loss = torch.stack(endpoint_losses).mean()
        consistency_loss = torch.stack(consistency_losses).mean()
        chamfer_loss = torch.stack(chamfer_losses).mean()
    else:
        line_loss = pred_lines.sum() * 0.0
        endpoint_loss = pred_endpoints.sum() * 0.0
        consistency_loss = pred_lines.sum() * 0.0
        chamfer_loss = pred_endpoints.sum() * 0.0

    total = (
        cls_loss
        + line_weight * line_loss
        + endpoint_weight * endpoint_loss
        + consistency_weight * consistency_loss
        + chamfer_weight * chamfer_loss
    )
    return {
        "loss": total,
        "cls_loss": cls_loss.detach(),
        "line_loss": line_loss.detach(),
        "endpoint_loss": endpoint_loss.detach(),
        "consistency_loss": consistency_loss.detach(),
        "chamfer_loss": chamfer_loss.detach(),
    }
