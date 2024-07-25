import random
from typing import List, Sequence

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms.functional import hflip, resize, resized_crop


@TRANSFORMS.register_module()
class FlipPoints(BaseTransform):

    def __init__(self,
                 flip_ratio_bev_horizontal: float = 0.0,
                 flip_ratio_bev_vertical: float = 0.0) -> None:
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def transform(self, results: dict) -> dict:
        if np.random.rand() < self.flip_ratio_bev_horizontal:
            results['points'].flip('horizontal')
        if np.random.rand() < self.flip_ratio_bev_vertical:
            results['points'].flip('vertical')

        if 'sweep_points' in results:
            if np.random.rand() < self.flip_ratio_bev_horizontal:
                results['sweep_points'].flip('horizontal')
            if np.random.rand() < self.flip_ratio_bev_vertical:
                results['sweep_points'].flip('vertical')

        return results


@TRANSFORMS.register_module()
class GlobalRotScaleTransPoints(BaseTransform):

    def __init__(self,
                 rot_range: List[float] = [-0.78539816, 0.78539816],
                 scale_ratio_range: List[float] = [0.95, 1.05],
                 translation_std: List[int] = [0, 0, 0]) -> None:
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'

        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = translation_std

    def transform(self, results: dict) -> dict:
        noise_rotation = np.random.uniform(self.rot_range[0],
                                           self.rot_range[1])
        results['points'].rotate(noise_rotation)
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        results['points'].scale(scale_factor)
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T
        results['points'].translate(trans_factor)

        if 'sweep_points' in results:
            noise_rotation = np.random.uniform(self.rot_range[0],
                                               self.rot_range[1])
            results['sweep_points'].rotate(noise_rotation)
            scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                             self.scale_ratio_range[1])
            results['sweep_points'].scale(scale_factor)
            translation_std = np.array(self.translation_std, dtype=np.float32)
            trans_factor = np.random.normal(scale=translation_std, size=3).T
            results['sweep_points'].translate(trans_factor)
        return results


@TRANSFORMS.register_module()
class ResizedCrop(BaseTransform):

    def __init__(self,
                 image_crop_size: Sequence[int] = (224, 416),
                 image_crop_range: Sequence[float] = (0.3, 1.0),
                 image_crop_ratio: Sequence[float] = (14.0 / 9.0, 17.0 / 9.0),
                 crop_center: bool = False) -> None:
        self.crop_size = image_crop_size
        self.crop_range = image_crop_range
        self.crop_ratio = image_crop_ratio
        self.crop_center = crop_center

    def transform(self, results: dict) -> dict:
        images = results['imgs']
        superpixels = results['superpixels'].unsqueeze(1)
        pairing_points = results['pairing_points']
        pairing_images = results['pairing_images']

        imgs = torch.empty(
            (images.shape[0], 3) + tuple(self.crop_size), dtype=torch.float32)
        sps = torch.empty(
            (images.shape[0], ) + tuple(self.crop_size), dtype=torch.uint8)
        pairing_points_out = np.empty(0, dtype=np.int64)
        pairing_images_out = np.empty((0, 3), dtype=np.int64)

        if 'sweep_points' in results:
            sweep_pairing_points = results['sweep_pairing_points']
            sweep_pairing_images = results['sweep_pairing_images']
            sweep_pairing_points_out = np.empty(0, dtype=np.int64)
            sweep_pairing_images_out = np.empty((0, 3), dtype=np.int64)

        if self.crop_center:
            pairing_points_out = pairing_points

            if 'sweep_points' in results:
                sweep_pairing_points_out = sweep_pairing_points

            _, _, h, w = images.shape
            for id, img in enumerate(images):
                mask = pairing_images[:, 0] == id
                p2 = pairing_images[mask]
                p2 = np.round(
                    np.multiply(
                        p2,
                        [1.0, self.crop_size[0] / h, self.crop_size[1] / w
                         ])).astype(np.int64)
                imgs[id] = resize(img, self.crop_size,
                                  InterpolationMode.BILINEAR)
                sps[id] = resize(superpixels[id], self.crop_size,
                                 InterpolationMode.NEAREST)
                p2[:, 1] = np.clip(0, self.crop_size[0] - 1, p2[:, 1])
                p2[:, 2] = np.clip(0, self.crop_size[1] - 1, p2[:, 2])
                pairing_images_out = np.concatenate((pairing_images_out, p2))
                if 'sweep_points' in results:
                    mask = sweep_pairing_images[:, 0] == id
                    p2 = sweep_pairing_images[mask]
                    p2 = np.round(
                        np.multiply(p2, [
                            1.0, self.crop_size[0] / h, self.crop_size[1] / w
                        ])).astype(np.int64)
                    p2[:, 1] = np.clip(0, self.crop_size[0] - 1, p2[:, 1])
                    p2[:, 2] = np.clip(0, self.crop_size[1] - 1, p2[:, 2])
                    sweep_pairing_images_out = np.concatenate(
                        (sweep_pairing_images_out, p2))
        else:
            assert 'sweep_points' not in results
            for id, img in enumerate(images):
                successful = False
                mask = pairing_images[:, 0] == id
                P1 = pairing_points[mask]
                P2 = pairing_images[mask]
                while not successful:
                    i, j, h, w = RandomResizedCrop.get_params(
                        img, self.crop_range, self.crop_ratio)
                    p1 = P1.copy()
                    p2 = P2.copy()
                    p2 = np.round(
                        np.multiply(p2 - [0, i, j], [
                            1.0, self.crop_size[0] / h, self.crop_size[1] / w
                        ])).astype(np.int64)
                    valid_indexes_0 = np.logical_and(
                        p2[:, 1] < self.crop_size[0], p2[:, 1] >= 0)
                    valid_indexes_1 = np.logical_and(
                        p2[:, 2] < self.crop_size[1], p2[:, 2] >= 0)
                    valid_indexes = np.logical_and(valid_indexes_0,
                                                   valid_indexes_1)
                    sum_indexes = valid_indexes.sum()
                    len_indexes = len(valid_indexes)
                    if sum_indexes > 1024 or sum_indexes / len_indexes > 0.75:
                        successful = True
                imgs[id] = resized_crop(img, i, j, h, w, self.crop_size,
                                        InterpolationMode.BILINEAR)
                sps[id] = resized_crop(superpixels[id], i, j, h, w,
                                       self.crop_size,
                                       InterpolationMode.NEAREST)
                pairing_points_out = np.concatenate(
                    (pairing_points_out, p1[valid_indexes]))
                pairing_images_out = np.concatenate(
                    (pairing_images_out, p2[valid_indexes]))

        results['imgs'] = imgs
        results['superpixels'] = sps
        results['pairing_points'] = pairing_points_out
        results['pairing_images'] = pairing_images_out
        if 'sweep_points' in results:
            results['sweep_pairing_points'] = sweep_pairing_points_out
            results['sweep_pairing_images'] = sweep_pairing_images_out
        return results


@TRANSFORMS.register_module()
class FlipHorizontal(BaseTransform):

    def __init__(self, flip_ratio: float = 0.5) -> None:
        self.flip_ratio = flip_ratio

    def transform(self, results: dict) -> dict:
        images = results['imgs']
        superpixels = results['superpixels']
        pairing_images = results['pairing_images']
        if 'sweep_points' in results:
            sweep_pairing_images = results['sweep_pairing_images']

        w = images.shape[3]
        for i, img in enumerate(images):
            if random.random() < self.flip_ratio:
                images[i] = hflip(img)
                superpixels[i] = hflip(superpixels[i:i + 1])
                mask = pairing_images[:, 0] == i
                pairing_images[mask, 2] = w - 1 - pairing_images[mask, 2]
                if 'sweep_points' in results:
                    mask = sweep_pairing_images[:, 0] == i
                    sweep_pairing_images[
                        mask, 2] = w - 1 - sweep_pairing_images[mask, 2]
        results['imgs'] = images
        results['superpixels'] = superpixels
        results['pairing_images'] = pairing_images
        if 'sweep_points' in results:
            results['sweep_pairing_images'] = sweep_pairing_images
        return results
