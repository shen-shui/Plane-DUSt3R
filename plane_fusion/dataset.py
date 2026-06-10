from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def plane_to_xz_line(plane):
    plane = np.asarray(plane, dtype=np.float32).copy()
    norm = np.linalg.norm(plane[[0, 2]])
    if norm > 1e-6:
        plane /= norm
    return np.asarray([plane[0], plane[2], plane[3]], dtype=np.float32)


def canonical_endpoint_key(endpoints, decimals=3):
    endpoints = np.asarray(endpoints, dtype=np.float32)
    p0 = endpoints[:2]
    p1 = endpoints[2:]
    direct = np.r_[p0, p1]
    flipped = np.r_[p1, p0]
    value = direct if tuple(np.round(direct, decimals)) <= tuple(np.round(flipped, decimals)) else flipped
    return tuple(np.round(value, decimals))


def load_targets_from_candidate_npz(data):
    keep = data["keep_target"] > 0.5
    plane_target = data["plane_target"][keep]
    endpoint_target = data["endpoint_target"][keep]

    seen = {}
    for plane, endpoints in zip(plane_target, endpoint_target):
        key = canonical_endpoint_key(endpoints)
        if key not in seen:
            seen[key] = (plane_to_xz_line(plane), endpoints.astype(np.float32))

    if not seen:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    lines, endpoints = zip(*seen.values())
    return np.stack(lines).astype(np.float32), np.stack(endpoints).astype(np.float32)


class PlaneFusionCandidateDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(Path(data_dir).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = np.load(self.files[index], allow_pickle=True)
        target_lines, target_endpoints = load_targets_from_candidate_npz(data)
        return {
            "features": torch.from_numpy(data["features"]).float(),
            "target_lines": torch.from_numpy(target_lines).float(),
            "target_endpoints": torch.from_numpy(target_endpoints).float(),
            "path": str(self.files[index]),
        }


def collate_plane_fusion(batch):
    batch_size = len(batch)
    max_candidates = max(item["features"].shape[0] for item in batch)
    max_targets = max(item["target_lines"].shape[0] for item in batch)
    feat_dim = batch[0]["features"].shape[1]

    out = {
        "features": torch.zeros(batch_size, max_candidates, feat_dim),
        "candidate_mask": torch.zeros(batch_size, max_candidates, dtype=torch.bool),
        "target_lines": torch.zeros(batch_size, max_targets, 3),
        "target_endpoints": torch.zeros(batch_size, max_targets, 4),
        "target_mask": torch.zeros(batch_size, max_targets, dtype=torch.bool),
        "path": [item["path"] for item in batch],
    }
    for i, item in enumerate(batch):
        n = item["features"].shape[0]
        m = item["target_lines"].shape[0]
        out["features"][i, :n] = item["features"]
        out["candidate_mask"][i, :n] = True
        if m:
            out["target_lines"][i, :m] = item["target_lines"]
            out["target_endpoints"][i, :m] = item["target_endpoints"]
            out["target_mask"][i, :m] = True
    return out
