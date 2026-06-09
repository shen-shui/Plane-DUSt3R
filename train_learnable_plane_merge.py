import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from learnable_plane_merge.dataset import MergeCandidateDataset, collate_merge_candidates
from learnable_plane_merge.features import FEATURE_DIM
from learnable_plane_merge.model import LearnablePlaneMerge, learnable_merge_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train learnable Plane-DUSt3R wall merge scorer.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="checkpoints/learnable_plane_merge.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--keep_pos_weight", type=float, default=1.0)
    parser.add_argument("--regression_weight", type=float, default=0.25)
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
    totals = {"loss": 0.0, "keep_loss": 0.0, "plane_loss": 0.0, "endpoint_loss": 0.0}
    steps = 0
    for batch in loader:
        batch = move_to_device(batch, args.device)
        with torch.set_grad_enabled(train):
            outputs = model(batch["features"], batch["mask"])
            losses = learnable_merge_loss(
                outputs,
                batch,
                keep_pos_weight=args.keep_pos_weight,
                regression_weight=args.regression_weight,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
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

    dataset = MergeCandidateDataset(args.data_dir)
    val_len = max(1, int(len(dataset) * args.val_ratio)) if len(dataset) > 1 else 0
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(args.seed)
    if val_len:
        train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)
    else:
        train_set, val_set = dataset, None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_merge_candidates,
    )
    val_loader = (
        DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_merge_candidates)
        if val_set is not None
        else None
    )

    model = LearnablePlaneMerge(
        input_dim=FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, args, train=True)
        if val_loader is not None:
            val_stats = run_epoch(model, val_loader, optimizer, args, train=False)
            score = val_stats["loss"]
        else:
            val_stats = train_stats
            score = train_stats["loss"]
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_stats['loss']:.4f} "
            f"val_loss={val_stats['loss']:.4f} "
            f"val_keep={val_stats['keep_loss']:.4f} "
            f"val_endpoint={val_stats['endpoint_loss']:.4f}"
        )
        if score < best_val:
            best_val = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "input_dim": FEATURE_DIM,
                    "best_val": best_val,
                },
                output_path,
            )
    print(f"saved best checkpoint to {output_path}")


if __name__ == "__main__":
    main()
