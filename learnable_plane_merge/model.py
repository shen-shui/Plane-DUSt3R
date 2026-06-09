import torch
from torch import nn


class LearnablePlaneMerge(nn.Module):
    """Score and refine global wall candidates after geometric plane merge."""

    def __init__(
        self,
        input_dim=19,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.keep_head = nn.Linear(hidden_dim, 1)
        self.plane_delta_head = nn.Linear(hidden_dim, 4)
        self.endpoint_delta_head = nn.Linear(hidden_dim, 4)

    def forward(self, features, mask=None):
        """Run candidate scoring/refinement.

        Args:
            features: Tensor [B, N, C].
            mask: Bool tensor [B, N], True for valid candidates.
        """
        padding_mask = None if mask is None else ~mask.bool()
        x = self.input_proj(features)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return {
            "keep_logits": self.keep_head(x).squeeze(-1),
            "plane_delta": self.plane_delta_head(x),
            "endpoint_delta": self.endpoint_delta_head(x),
        }


def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom


def learnable_merge_loss(outputs, batch, keep_pos_weight=1.0, regression_weight=0.25):
    valid = batch["mask"].bool()
    keep_target = batch["keep_target"].float()
    pos_weight = torch.as_tensor(keep_pos_weight, device=keep_target.device, dtype=keep_target.dtype)
    bce = nn.functional.binary_cross_entropy_with_logits(
        outputs["keep_logits"][valid],
        keep_target[valid],
        pos_weight=pos_weight,
    )

    matched = valid & (keep_target > 0.5)
    if matched.any():
        plane_pred = batch["base_plane"] + outputs["plane_delta"]
        endpoint_pred = batch["base_endpoints"] + outputs["endpoint_delta"]
        plane_loss = nn.functional.smooth_l1_loss(
            plane_pred[matched],
            batch["plane_target"][matched],
        )
        endpoint_loss = nn.functional.smooth_l1_loss(
            endpoint_pred[matched],
            batch["endpoint_target"][matched],
        )
    else:
        plane_loss = outputs["plane_delta"].sum() * 0.0
        endpoint_loss = outputs["endpoint_delta"].sum() * 0.0

    total = bce + regression_weight * (plane_loss + endpoint_loss)
    return {
        "loss": total,
        "keep_loss": bce.detach(),
        "plane_loss": plane_loss.detach(),
        "endpoint_loss": endpoint_loss.detach(),
    }
