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


def plane_fusion_loss(outputs, batch, line_weight=1.0, endpoint_weight=1.0, no_object_weight=0.1):
    logits = outputs["logits"]
    pred_lines = outputs["line"]
    pred_endpoints = outputs["endpoints"]
    target_lines = batch["target_lines"]
    target_endpoints = batch["target_endpoints"]
    target_mask = batch["target_mask"]

    cls_targets = torch.zeros_like(logits)
    line_losses = []
    endpoint_losses = []

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
            - logits[b].sigmoid().unsqueeze(-1)
        )
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

    pos_weight = torch.ones_like(logits)
    pos_weight[cls_targets < 0.5] = no_object_weight
    cls_loss = F.binary_cross_entropy_with_logits(logits, cls_targets, weight=pos_weight)
    if line_losses:
        line_loss = torch.stack(line_losses).mean()
        endpoint_loss = torch.stack(endpoint_losses).mean()
    else:
        line_loss = pred_lines.sum() * 0.0
        endpoint_loss = pred_endpoints.sum() * 0.0

    total = cls_loss + line_weight * line_loss + endpoint_weight * endpoint_loss
    return {
        "loss": total,
        "cls_loss": cls_loss.detach(),
        "line_loss": line_loss.detach(),
        "endpoint_loss": endpoint_loss.detach(),
    }
