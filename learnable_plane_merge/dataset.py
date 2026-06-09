from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MergeCandidateDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(Path(data_dir).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz samples found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = np.load(self.files[index], allow_pickle=True)
        return {
            "features": torch.from_numpy(data["features"]).float(),
            "keep_target": torch.from_numpy(data["keep_target"]).float(),
            "base_plane": torch.from_numpy(data["base_plane"]).float(),
            "base_endpoints": torch.from_numpy(data["base_endpoints"]).float(),
            "plane_target": torch.from_numpy(data["plane_target"]).float(),
            "endpoint_target": torch.from_numpy(data["endpoint_target"]).float(),
            "path": str(self.files[index]),
        }


def collate_merge_candidates(batch):
    max_walls = max(item["features"].shape[0] for item in batch)
    feat_dim = batch[0]["features"].shape[1]
    batch_size = len(batch)

    output = {
        "features": torch.zeros(batch_size, max_walls, feat_dim),
        "keep_target": torch.zeros(batch_size, max_walls),
        "base_plane": torch.zeros(batch_size, max_walls, 4),
        "base_endpoints": torch.zeros(batch_size, max_walls, 4),
        "plane_target": torch.zeros(batch_size, max_walls, 4),
        "endpoint_target": torch.zeros(batch_size, max_walls, 4),
        "mask": torch.zeros(batch_size, max_walls, dtype=torch.bool),
        "path": [item["path"] for item in batch],
    }
    for i, item in enumerate(batch):
        n = item["features"].shape[0]
        output["features"][i, :n] = item["features"]
        output["keep_target"][i, :n] = item["keep_target"]
        output["base_plane"][i, :n] = item["base_plane"]
        output["base_endpoints"][i, :n] = item["base_endpoints"]
        output["plane_target"][i, :n] = item["plane_target"]
        output["endpoint_target"][i, :n] = item["endpoint_target"]
        output["mask"][i, :n] = True
    return output
