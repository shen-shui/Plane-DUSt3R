import numpy as np


FEATURE_DIM = 19


def normalize_plane(plane):
    plane = np.asarray(plane, dtype=np.float64).copy()
    norm = np.linalg.norm(plane[:3])
    if norm > 0:
        plane /= norm
    return plane


def wall_info_to_feature(info, scale=1.0):
    plane = normalize_plane(info["pparam"])
    plane = plane.copy()
    plane[3] *= scale

    left = np.asarray(info.get("left_endpoint"), dtype=np.float64)
    right = np.asarray(info.get("right_endpoint"), dtype=np.float64)
    if left.shape[0] >= 3:
        left_xz = left[[0, 2]] * scale
    else:
        left_xz = np.zeros(2, dtype=np.float64)
    if right.shape[0] >= 3:
        right_xz = right[[0, 2]] * scale
    else:
        right_xz = np.zeros(2, dtype=np.float64)

    center = (left_xz + right_xz) / 2
    vector = right_xz - left_xz
    length = np.linalg.norm(vector)
    direction = vector / max(length, 1e-6)
    support_views = info.get("support_views", [])
    line_count = float(info.get("line_count", 1))
    support_count = float(len(set(support_views)))
    has_pre = float(info.get("pre") is not None)
    has_next = float(info.get("next") is not None)

    feature = np.asarray(
        [
            *plane.tolist(),
            *left_xz.tolist(),
            *right_xz.tolist(),
            *center.tolist(),
            *direction.tolist(),
            length,
            line_count,
            support_count,
            has_pre,
            has_next,
            float(info.get("index", 0)),
            1.0,
        ],
        dtype=np.float32,
    )
    if feature.shape[0] != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM} features, got {feature.shape[0]}")
    return feature


def wall_info_to_base_targets(info, scale=1.0):
    plane = normalize_plane(info["pparam"])
    plane = plane.copy()
    plane[3] *= scale
    left = np.asarray(info.get("left_endpoint"), dtype=np.float64)[[0, 2]] * scale
    right = np.asarray(info.get("right_endpoint"), dtype=np.float64)[[0, 2]] * scale
    endpoints = np.asarray([left[0], left[1], right[0], right[1]], dtype=np.float32)
    return plane.astype(np.float32), endpoints
