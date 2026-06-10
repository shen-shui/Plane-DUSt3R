import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from plane_fusion.dataset import PlaneFusionCandidateDataset, collate_plane_fusion
from plane_fusion.losses import plane_fusion_loss
from plane_fusion.model import PlaneFusionDETR


def parse_args():
    parser = argparse.ArgumentParser(description="Train DETR-style plane fusion for 2D wall prediction.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="checkpoints/plane_fusion_detr.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_queries", type=int, default=16)
    parser.add_argument("--line_weight", type=float, default=1.0)
    parser.add_argument("--endpoint_weight", type=float, default=1.0)
    parser.add_argument("--no_object_weight", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def run_epoch(model, loader, optimizer, args, train=True):
    model.train(train)
    totals = {"loss": 0.0, "cls_loss": 0.0, "line_loss": 0.0, "endpoint_loss": 0.0}
    steps = 0
    for batch in loader:
        batch = move_to_device(batch, args.device)
        with torch.set_grad_enabled(train):
            outputs = model(batch["features"], batch["candidate_mask"])
            losses = plane_fusion_loss(
                outputs,
                batch,
                line_weight=args.line_weight,
                endpoint_weight=args.endpoint_weight,
                no_object_weight=args.no_object_weight,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        for key in totals:
            totals[key] += float(losses[key])
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    dataset = PlaneFusionCandidateDataset(args.data_dir)
    input_dim = dataset[0]["features"].shape[1]
    val_len = max(1, int(len(dataset) * args.val_ratio)) if len(dataset) > 1 else 0
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator) if val_len else (dataset, None)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_plane_fusion)
    val_loader = (
        DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_plane_fusion)
        if val_set is not None
        else None
    )

    model = PlaneFusionDETR(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        num_queries=args.num_queries,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, args, train=True)
        val_stats = run_epoch(model, val_loader, optimizer, args, train=False) if val_loader else train_stats
        print(
            f"epoch {epoch:03d} "
            f"train={train_stats['loss']:.4f} "
            f"val={val_stats['loss']:.4f} "
            f"cls={val_stats['cls_loss']:.4f} "
            f"line={val_stats['line_loss']:.4f} "
            f"end={val_stats['endpoint_loss']:.4f}"
        )
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "input_dim": input_dim,
                    "best_val": best_val,
                },
                output,
            )
    print(f"saved best checkpoint to {output}")


if __name__ == "__main__":
    main()
