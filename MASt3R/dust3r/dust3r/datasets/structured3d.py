# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import itertools
import os
import os.path as osp

import cv2
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class Structured3D(BaseStereoViewDataset):
    def __init__(self, *args, ROOT, split='train', num_samples=None, **kwargs):
        self.ROOT = ROOT
        self.num_samples = num_samples
        super().__init__(*args, split=split, **kwargs)
        self.rooms = self._find_rooms()
        self.pairs = self._make_pairs()

    def __len__(self):
        return len(self.pairs)

    def _scene_in_split(self, scene_name):
        scene_id = int(scene_name.split('_')[-1])
        if self.split == 'train':
            return scene_id <= 2999
        if self.split in ('val', 'validation'):
            return 3000 <= scene_id <= 3249
        if self.split == 'test':
            return scene_id >= 3250
        raise ValueError(f'bad split={self.split}')

    def _find_rooms(self):
        rooms = []
        for scene_name in sorted(os.listdir(self.ROOT)):
            scene_dir = osp.join(self.ROOT, scene_name)
            if not osp.isdir(scene_dir) or not scene_name.startswith('scene_') or not self._scene_in_split(scene_name):
                continue

            rendering_dir = osp.join(scene_dir, '2D_rendering')
            if not osp.isdir(rendering_dir):
                continue

            for room_name in sorted(os.listdir(rendering_dir)):
                full_dir = osp.join(rendering_dir, room_name, 'perspective', 'full')
                if not osp.isdir(full_dir):
                    continue

                positions = []
                for pos_name in sorted(os.listdir(full_dir), key=lambda x: int(x) if x.isdigit() else x):
                    pos_dir = osp.join(full_dir, pos_name)
                    required = ['rgb_rawlight.png', 'plane_depth.png', 'camera_pose.txt']
                    if osp.isdir(pos_dir) and all(osp.isfile(osp.join(pos_dir, f)) for f in required):
                        positions.append(pos_dir)
                if len(positions) >= 2:
                    rooms.append(positions)
        return rooms

    def _make_pairs(self):
        pairs = []
        for positions in self.rooms:
            pairs.extend(itertools.combinations(positions, 2))
        if self.num_samples is not None:
            pairs = pairs[:self.num_samples]
        return pairs

    def _read_pose_and_intrinsics(self, pose_path, image_shape):
        nums = np.loadtxt(pose_path, dtype=np.float32).reshape(-1)
        if nums.size == 16:
            return nums.reshape(4, 4).astype(np.float32), self._default_intrinsics(image_shape)
        if nums.size == 12:
            pose = np.eye(4, dtype=np.float32)
            pose[:3] = nums.reshape(3, 4)
            return pose, self._default_intrinsics(image_shape)

        if nums.size < 9:
            raise ValueError(f'Unsupported camera pose format in {pose_path}: {nums}')

        eye = nums[:3].astype(np.float32)
        forward = nums[3:6].astype(np.float32)
        up = nums[6:9].astype(np.float32)
        x_fov = float(nums[9]) if nums.size > 9 else None
        y_fov = float(nums[10]) if nums.size > 10 else None

        forward = forward / np.clip(np.linalg.norm(forward), 1e-8, None)
        up = up / np.clip(np.linalg.norm(up), 1e-8, None)
        right = np.cross(forward, up)
        right = right / np.clip(np.linalg.norm(right), 1e-8, None)
        down = np.cross(forward, right)
        down = down / np.clip(np.linalg.norm(down), 1e-8, None)

        pose = np.eye(4, dtype=np.float32)
        pose[:3, 0] = right
        pose[:3, 1] = down
        pose[:3, 2] = forward
        pose[:3, 3] = eye / 1000.0

        return pose, self._intrinsics_from_fov(image_shape, x_fov, y_fov)

    def _default_intrinsics(self, image_shape):
        height, width = image_shape
        scale_x = width / 1280.0
        scale_y = height / 720.0
        return np.array([
            [762.0 * scale_x, 0.0, width / 2.0],
            [0.0, 762.0 * scale_y, height / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def _intrinsics_from_fov(self, image_shape, x_fov, y_fov):
        if x_fov is None or y_fov is None:
            return self._default_intrinsics(image_shape)

        height, width = image_shape
        # Structured3D stores FOV in radians in recent releases; fall back gracefully if degrees are found.
        if x_fov > np.pi:
            x_fov = np.deg2rad(x_fov)
        if y_fov > np.pi:
            y_fov = np.deg2rad(y_fov)
        fx = width / (2.0 * np.tan(x_fov / 2.0))
        fy = height / (2.0 * np.tan(y_fov / 2.0))
        return np.array([
            [fx, 0.0, width / 2.0],
            [0.0, fy, height / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def _get_views(self, idx, resolution, rng):
        view_dirs = self.pairs[idx]
        views = []

        for view_dir in view_dirs:
            rgb_path = osp.join(view_dir, 'rgb_rawlight.png')
            depth_path = osp.join(view_dir, 'plane_depth.png')
            pose_path = osp.join(view_dir, 'camera_pose.txt')

            rgb_image = imread_cv2(rgb_path)
            depthmap = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            camera_pose, intrinsics = self._read_pose_and_intrinsics(pose_path, depthmap.shape)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=rgb_path)

            semantic_mask = (depthmap <= 0).astype(np.float32)

            rel = osp.relpath(view_dir, self.ROOT)
            scene_name, _, room_name, _, _, pos_name = rel.split(osp.sep)
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                semantic_mask=semantic_mask,
                dataset='Structured3D',
                label=f'{scene_name}_{room_name}',
                instance=pos_name,
            ))

        return views
