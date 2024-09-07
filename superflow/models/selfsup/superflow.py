from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import (ForwardResults,
                                                  OptSampleList, SampleList)
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModel
from torch import Tensor


class ConstractiveLoss(nn.Module):

    def __init__(self, temperature: float) -> None:
        super(ConstractiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, k: Tensor, q: Tensor) -> Tensor:
        logits = torch.mm(k, q.transpose(1, 0))
        target = torch.arange(k.shape[0], device=k.device).long()
        out = torch.div(logits, self.temperature)
        out = out.contiguous()
        loss = self.criterion(out, target)
        return loss


@MODELS.register_module()
class SuperFlow(BaseModel):

    def __init__(self,
                 backbone_3d: ConfigType,
                 head_3d: ConfigType,
                 backbone_2d: ConfigType,
                 head_2d: ConfigType,
                 superpixel_size: int,
                 temperature: float = None,
                 data_preprocessor: OptConfigType = None,
                 train_cfg: ConfigType = dict(),
                 init_cfg: OptMultiConfig = None):
        super(SuperFlow, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone_3d = MODELS.build(backbone_3d)
        self.head_3d = MODELS.build(head_3d)
        self.backbone_2d = MODELS.build(backbone_2d)
        self.head_2d = MODELS.build(head_2d)
        self.superpixel_size = superpixel_size
        self.contrastive_losses = ConstractiveLoss(temperature)
        self.train_cfg = train_cfg

    def extract_3d_feature(self, feat_dict: dict) -> Tensor:
        feat_dict = self.backbone_3d(feat_dict)
        features = self.head_3d(feat_dict)['logits']
        features = F.normalize(features, p=2, dim=1)
        return features

    def extract_2d_feature(self, images: Tensor) -> Tensor:
        features = self.backbone_2d(images)
        features = self.head_2d(features)
        features = F.normalize(features, p=2, dim=1)
        return features

    def loss(self, inputs: dict,
             data_samples: SampleList) -> Dict[str, Tensor]:

        feat_dict = inputs['voxels']
        prev_feat_dict = inputs['prev_voxels']
        next_feat_dict = inputs['next_voxels']
        sweep_feat_dict = inputs['sweep_voxels']

        # forward
        features_3d = self.extract_3d_feature(feat_dict)
        prev_features_3d = self.extract_3d_feature(prev_feat_dict)
        next_features_3d = self.extract_3d_feature(next_feat_dict)
        sweep_features_3d = self.extract_3d_feature(sweep_feat_dict)

        features_2d = self.extract_2d_feature(inputs['imgs'])
        prev_features_2d = self.extract_2d_feature(inputs['prev_imgs'])
        next_features_2d = self.extract_2d_feature(inputs['next_imgs'])

        superpixels = []
        pairing_images = []
        pairing_points = []

        prev_superpixels = []
        prev_pairing_images = []
        prev_pairing_points = []

        next_superpixels = []
        next_pairing_images = []
        next_pairing_points = []

        sweep_pairing_images = []
        sweep_pairing_points = []

        offset = 0
        prev_offset = 0
        next_offset = 0
        sweep_offset = 0

        for i, data_sample in enumerate(data_samples):
            superpixel = data_sample.gt_pts_seg.superpixels
            pairing_image = data_sample.gt_pts_seg.pairing_images
            pairing_image[:, 0] += i * superpixel.shape[0]
            pairing_point = data_sample.gt_pts_seg.pairing_points
            inverse_map = feat_dict['point2voxel_maps'][i]
            pairing_point = inverse_map[pairing_point].long() + offset
            offset += feat_dict['voxel_inds'][i].shape[0]

            prev_superpixel = data_sample.gt_pts_seg.prev_superpixels
            prev_pairing_image = data_sample.gt_pts_seg.prev_pairing_images
            prev_pairing_image[:, 0] += i * prev_superpixel.shape[0]
            prev_pairing_point = data_sample.gt_pts_seg.prev_pairing_points
            prev_inverse_map = prev_feat_dict['point2voxel_maps'][i]
            prev_pairing_point = prev_inverse_map[prev_pairing_point].long(
            ) + prev_offset
            prev_offset += prev_feat_dict['voxel_inds'][i].shape[0]

            next_superpixel = data_sample.gt_pts_seg.next_superpixels
            next_pairing_image = data_sample.gt_pts_seg.next_pairing_images
            next_pairing_image[:, 0] += i * next_superpixel.shape[0]
            next_pairing_point = data_sample.gt_pts_seg.next_pairing_points
            next_inverse_map = next_feat_dict['point2voxel_maps'][i]
            next_pairing_point = next_inverse_map[next_pairing_point].long(
            ) + next_offset
            next_offset += next_feat_dict['voxel_inds'][i].shape[0]

            sweep_pairing_image = data_sample.gt_pts_seg.sweep_pairing_images
            sweep_pairing_image[:, 0] += i * superpixel.shape[0]
            sweep_pairing_point = data_sample.gt_pts_seg.sweep_pairing_points
            sweep_inverse_map = sweep_feat_dict['point2voxel_maps'][i]
            sweep_pairing_point = sweep_inverse_map[sweep_pairing_point].long(
            ) + sweep_offset
            sweep_offset += sweep_feat_dict['voxel_inds'][i].shape[0]

            superpixels.append(superpixel)
            pairing_images.append(pairing_image)
            pairing_points.append(pairing_point)

            prev_superpixels.append(prev_superpixel)
            prev_pairing_images.append(prev_pairing_image)
            prev_pairing_points.append(prev_pairing_point)

            next_superpixels.append(next_superpixel)
            next_pairing_images.append(next_pairing_image)
            next_pairing_points.append(next_pairing_point)

            sweep_pairing_images.append(sweep_pairing_image)
            sweep_pairing_points.append(sweep_pairing_point)

        superpixels = torch.cat(superpixels)
        pairing_images = torch.cat(pairing_images)
        pairing_points = torch.cat(pairing_points)

        prev_superpixels = torch.cat(prev_superpixels)
        prev_pairing_images = torch.cat(prev_pairing_images)
        prev_pairing_points = torch.cat(prev_pairing_points)

        next_superpixels = torch.cat(next_superpixels)
        next_pairing_images = torch.cat(next_pairing_images)
        next_pairing_points = torch.cat(next_pairing_points)

        sweep_pairing_images = torch.cat(sweep_pairing_images)
        sweep_pairing_points = torch.cat(sweep_pairing_points)

        interleave = torch.arange(
            0,
            features_2d.shape[0] * self.superpixel_size,
            self.superpixel_size,
            device=features_2d.device)
        superpixels = interleave[:, None, None] + superpixels
        prev_superpixels = interleave[:, None, None] + prev_superpixels
        next_superpixels = interleave[:, None, None] + next_superpixels

        m = tuple(pairing_images.cpu().T.long())
        superpixels_I = superpixels.flatten()
        idx_P = torch.arange(
            pairing_points.shape[0], device=features_2d.device)
        total_pixels = superpixels_I.shape[0]
        idx_I = torch.arange(total_pixels, device=features_2d.device)

        prev_m = tuple(prev_pairing_images.cpu().T.long())
        prev_superpixels_I = prev_superpixels.flatten()
        prev_idx_P = torch.arange(
            prev_pairing_points.shape[0], device=prev_features_2d.device)
        prev_total_pixels = prev_superpixels_I.shape[0]
        prev_idx_I = torch.arange(
            prev_total_pixels, device=prev_features_2d.device)

        next_m = tuple(next_pairing_images.cpu().T.long())
        next_superpixels_I = next_superpixels.flatten()
        next_idx_P = torch.arange(
            next_pairing_points.shape[0], device=next_features_2d.device)
        next_total_pixels = next_superpixels_I.shape[0]
        next_idx_I = torch.arange(
            next_total_pixels, device=next_features_2d.device)

        sweep_m = tuple(sweep_pairing_images.cpu().T.long())
        sweep_idx_P = torch.arange(
            sweep_pairing_points.shape[0], device=features_2d.device)

        with torch.no_grad():
            one_hot_P = torch.sparse_coo_tensor(
                torch.stack((superpixels[m], idx_P), dim=0),
                torch.ones(pairing_points.shape[0], device=features_2d.device),
                (superpixels.shape[0] * self.superpixel_size,
                 pairing_points.shape[0]))
            one_hot_I = torch.sparse_coo_tensor(
                torch.stack((superpixels_I, idx_I), dim=0),
                torch.ones(total_pixels, device=features_2d.device),
                (superpixels.shape[0] * self.superpixel_size, total_pixels))

            prev_one_hot_P = torch.sparse_coo_tensor(
                torch.stack((prev_superpixels[prev_m], prev_idx_P), dim=0),
                torch.ones(
                    prev_pairing_points.shape[0],
                    device=prev_features_2d.device),
                (prev_superpixels.shape[0] * self.superpixel_size,
                 prev_pairing_points.shape[0]))
            prev_one_hot_I = torch.sparse_coo_tensor(
                torch.stack((prev_superpixels_I, prev_idx_I), dim=0),
                torch.ones(prev_total_pixels, device=prev_features_2d.device),
                (prev_superpixels.shape[0] * self.superpixel_size,
                 prev_total_pixels))

            next_one_hot_P = torch.sparse_coo_tensor(
                torch.stack((next_superpixels[next_m], next_idx_P), dim=0),
                torch.ones(
                    next_pairing_points.shape[0],
                    device=next_features_2d.device),
                (next_superpixels.shape[0] * self.superpixel_size,
                 next_pairing_points.shape[0]))
            next_one_hot_I = torch.sparse_coo_tensor(
                torch.stack((next_superpixels_I, next_idx_I), dim=0),
                torch.ones(next_total_pixels, device=next_features_2d.device),
                (next_superpixels.shape[0] * self.superpixel_size,
                 next_total_pixels))

            sweep_one_hot_P = torch.sparse_coo_tensor(
                torch.stack((superpixels[sweep_m], sweep_idx_P), dim=0),
                torch.ones(
                    sweep_pairing_points.shape[0], device=features_2d.device),
                (superpixels.shape[0] * self.superpixel_size,
                 sweep_pairing_points.shape[0]))

        k = one_hot_P @ features_3d[pairing_points]
        k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)
        q = one_hot_I @ features_2d.permute(0, 2, 3, 1).flatten(0, 2)
        q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)

        prev_k = prev_one_hot_P @ prev_features_3d[prev_pairing_points]
        prev_k = prev_k / (
            torch.sparse.sum(prev_one_hot_P, 1).to_dense()[:, None] + 1e-6)
        prev_q = prev_one_hot_I @ prev_features_2d.permute(0, 2, 3, 1).flatten(
            0, 2)
        prev_q = prev_q / (
            torch.sparse.sum(prev_one_hot_I, 1).to_dense()[:, None] + 1e-6)

        next_k = next_one_hot_P @ next_features_3d[next_pairing_points]
        next_k = next_k / (
            torch.sparse.sum(next_one_hot_P, 1).to_dense()[:, None] + 1e-6)
        next_q = next_one_hot_I @ next_features_2d.permute(0, 2, 3, 1).flatten(
            0, 2)
        next_q = next_q / (
            torch.sparse.sum(next_one_hot_I, 1).to_dense()[:, None] + 1e-6)

        sweep_k = sweep_one_hot_P @ sweep_features_3d[sweep_pairing_points]
        sweep_k = sweep_k / (
            torch.sparse.sum(sweep_one_hot_P, 1).to_dense()[:, None] + 1e-6)

        mask = torch.where(k[:, 0] != 0)
        valid_k = k[mask]
        valid_q = q[mask]
        sweep_valid_k = sweep_k[mask]

        mask = torch.where(prev_k[:, 0] != 0)
        prev_valid_k = prev_k[mask]
        prev_valid_q = prev_q[mask]

        mask = torch.where(next_k[:, 0] != 0)
        next_valid_k = next_k[mask]
        next_valid_q = next_q[mask]

        loss = dict()
        loss['loss_spatial'] = (
            self.contrastive_losses(valid_k, valid_q) +
            self.contrastive_losses(prev_valid_k, prev_valid_q) +
            self.contrastive_losses(next_valid_k, next_valid_q)) / 3.0

        loss['loss_d2s'] = self.contrastive_losses(valid_k, sweep_valid_k)

        mask1 = torch.where((k[:, 0] != 0) & (prev_k[:, 0] != 0))
        mask2 = torch.where((k[:, 0] != 0) & (next_k[:, 0] != 0))
        loss['loss_temporal'] = (
            self.contrastive_losses(k[mask1], prev_k[mask1]) +
            self.contrastive_losses(k[mask2], next_k[mask2])) / 2.0

        return loss

    def forward(self,
                inputs: dict,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        if mode == 'loss':
            return self.loss(inputs, data_samples)
