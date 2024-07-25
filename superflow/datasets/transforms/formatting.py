from typing import Sequence

import torch
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import Det3DDataSample, PointData
from mmdet3d.structures.points import BasePoints


@TRANSFORMS.register_module()
class SuperflowInputs(BaseTransform):

    def __init__(self, keys: Sequence[str]) -> None:
        self.keys = keys

    def transform(self, results: dict) -> dict:
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor
        if 'prev_points' in results:
            if isinstance(results['prev_points'], BasePoints):
                results['prev_points'] = results['prev_points'].tensor
        if 'next_points' in results:
            if isinstance(results['next_points'], BasePoints):
                results['next_points'] = results['next_points'].tensor
        if 'sweep_points' in results:
            if isinstance(results['sweep_points'], BasePoints):
                results['sweep_points'] = results['sweep_points'].tensor

        if 'pairing_points' in results:
            results['pairing_points'] = torch.tensor(results['pairing_points'])
        if 'prev_pairing_points' in results:
            results['prev_pairing_points'] = torch.tensor(
                results['prev_pairing_points'])
        if 'next_pairing_points' in results:
            results['next_pairing_points'] = torch.tensor(
                results['next_pairing_points'])
        if 'sweep_pairing_points' in results:
            results['sweep_pairing_points'] = torch.tensor(
                results['sweep_pairing_points'])

        if 'pairing_images' in results:
            results['pairing_images'] = torch.tensor(results['pairing_images'])
        if 'prev_pairing_images' in results:
            results['prev_pairing_images'] = torch.tensor(
                results['prev_pairing_images'])
        if 'next_pairing_images' in results:
            results['next_pairing_images'] = torch.tensor(
                results['next_pairing_images'])
        if 'sweep_pairing_images' in results:
            results['sweep_pairing_images'] = torch.tensor(
                results['sweep_pairing_images'])

        data_sample = Det3DDataSample()
        gt_pts_seg = PointData()

        inputs = {}
        for key in self.keys:
            if key in ('points', 'imgs', 'prev_points', 'prev_imgs',
                       'next_points', 'next_imgs', 'sweep_points'):
                inputs[key] = results[key]
            elif key in ('pairing_points', 'pairing_images', 'superpixels',
                         'superpoints', 'prev_pairing_points',
                         'prev_pairing_images', 'prev_superpixels',
                         'prev_superpoints', 'next_pairing_points',
                         'next_pairing_images', 'next_superpixels',
                         'next_superpoints', 'sweep_pairing_points',
                         'sweep_pairing_images'):
                gt_pts_seg[key] = results[key]

        data_sample.gt_pts_seg = gt_pts_seg

        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs

        return packed_results
