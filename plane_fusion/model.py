import torch
from torch import nn


class PlaneFusionDETR(nn.Module):
    """Predict global 2D wall lines from multi-view plane candidate tokens."""

    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_heads=8,
        num_queries=16,
        dropout=0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
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
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.object_head = nn.Linear(hidden_dim, 1)
        self.line_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.endpoint_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, features, mask=None):
        padding_mask = None if mask is None else ~mask.bool()
        memory = self.encoder(self.input_proj(features), src_key_padding_mask=padding_mask)
        queries = self.query_embed.weight.unsqueeze(0).expand(features.shape[0], -1, -1)
        decoded = self.decoder(queries, memory, memory_key_padding_mask=padding_mask)
        line = self.line_head(decoded)
        line_norm = line[..., :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
        line = torch.cat([line[..., :2] / line_norm, line[..., 2:3]], dim=-1)
        return {
            "logits": self.object_head(decoded).squeeze(-1),
            "line": line,
            "endpoints": self.endpoint_head(decoded),
        }
