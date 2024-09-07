from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmengine.model import ImgDataPreprocessor
from mmengine.utils import is_seq_of
from torch import Tensor


@MODELS.register_module()
class SuperflowDataPreprocessor(ImgDataPreprocessor):

    def __init__(self,
                 voxel_size: Sequence[float],
                 voxel_type: str = 'cubic',
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 to_rgb: bool = False,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: bool = False) -> None:
        super(SuperflowDataPreprocessor, self).__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking)
        self._channel_conversion = to_rgb or bgr_to_rgb or rgb_to_bgr
        self.voxel_size = voxel_size
        self.voxel_type = voxel_type

    def forward(self, data: dict, training: bool = False) -> dict:
        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']
            batch_inputs['prev_points'] = inputs['prev_points']
            batch_inputs['next_points'] = inputs['next_points']
            batch_inputs['sweep_points'] = inputs['sweep_points']

            (voxel_dict, prev_voxel_dict, next_voxel_dict,
             sweep_voxel_dict) = self.voxelize(inputs['points'],
                                               inputs['prev_points'],
                                               inputs['next_points'],
                                               inputs['sweep_points'],
                                               data_samples)

            batch_inputs['voxels'] = voxel_dict
            batch_inputs['prev_voxels'] = prev_voxel_dict
            batch_inputs['next_voxels'] = next_voxel_dict
            batch_inputs['sweep_voxels'] = sweep_voxel_dict

        if 'imgs' in inputs:
            imgs = inputs['imgs']
            prev_imgs = inputs['prev_imgs']
            next_imgs = inputs['next_imgs']

            if data_samples is not None:
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample in data_samples:
                    data_sample.set_metainfo(
                        {'batch_input_shape': batch_input_shape})

            batch_inputs['imgs'] = imgs
            batch_inputs['prev_imgs'] = prev_imgs
            batch_inputs['next_imgs'] = next_imgs

        return {'inputs': batch_inputs, 'data_samples': data_samples}

    def preprocess_img(self, _batch_img: Tensor) -> Tensor:
        if self._channel_conversion:
            _batch_img = _batch_img[[2, 1, 0], ...]
        _batch_img = _batch_img.float()
        if self._enable_normalize:
            if self.mean.shape[0] == 3:
                assert _batch_img.dim() == 3 and _batch_img.shape[0] == 3
            _batch_img = (_batch_img - self.mean) / self.std
        return _batch_img

    def collate_data(self, data: dict) -> dict:
        data = self.cast_data(data)

        if 'imgs' in data['inputs']:
            _batch_imgs = data['inputs']['imgs']
            _batch_prev_imgs = data['inputs']['prev_imgs']
            _batch_next_imgs = data['inputs']['next_imgs']

            assert is_seq_of(_batch_imgs, Tensor)
            assert is_seq_of(_batch_prev_imgs, Tensor)
            assert is_seq_of(_batch_next_imgs, Tensor)

            batch_imgs = []
            batch_prev_imgs = []
            batch_next_imgs = []

            for _batch_img, _batch_prev_img, _batch_next_img in zip(
                    _batch_imgs, _batch_prev_imgs, _batch_next_imgs):
                _batch_img = [self.preprocess_img(_img) for _img in _batch_img]
                _batch_img = torch.stack(_batch_img, dim=0)
                batch_imgs.append(_batch_img)

                _batch_prev_img = [
                    self.preprocess_img(_img) for _img in _batch_prev_img
                ]
                _batch_prev_img = torch.stack(_batch_prev_img, dim=0)
                batch_prev_imgs.append(_batch_prev_img)

                _batch_next_img = [
                    self.preprocess_img(_img) for _img in _batch_next_img
                ]
                _batch_next_img = torch.stack(_batch_next_img, dim=0)
                batch_next_imgs.append(_batch_next_img)

            batch_imgs = torch.concat(batch_imgs, dim=0)
            batch_prev_imgs = torch.concat(batch_prev_imgs, dim=0)
            batch_next_imgs = torch.concat(batch_next_imgs, dim=0)

            data['inputs']['imgs'] = batch_imgs
            data['inputs']['prev_imgs'] = batch_prev_imgs
            data['inputs']['next_imgs'] = batch_next_imgs

        data.setdefault('data_samples', None)

        return data

    def voxelize_single(self, point: Tensor, voxel_size: Tensor,
                        index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.voxel_type == 'cubic':
            res_coors = torch.round(point[:, :3] / voxel_size).int()
        elif self.voxel_type == 'cylinder':
            rho = torch.sqrt(point[:, 0]**2 + point[:, 1]**2)
            phi = torch.atan2(point[:, 1], point[:, 0]) * 180 / np.pi
            polar_res = torch.stack((rho, phi, point[:, 2]), dim=1)
            res_coors = torch.round(polar_res[:, :3] / voxel_size).int()

        res_coors -= res_coors.min(0)[0]

        res_coors_numpy = res_coors.cpu().numpy()
        inds, point2voxel_map = self.sparse_quantize(
            res_coors_numpy, return_index=True, return_inverse=True)
        point2voxel_map = torch.from_numpy(point2voxel_map).cuda()
        inds = torch.from_numpy(inds).cuda()
        res_voxel_coors = res_coors[inds]
        res_voxels = point[inds]
        res_voxel_coors = F.pad(
            res_voxel_coors, (0, 1), mode='constant', value=index)

        return res_voxels, res_voxel_coors, point2voxel_map, inds

    @torch.no_grad()
    def voxelize(
        self, points: List[Tensor], prev_points: List[Tensor],
        next_points: List[Tensor], sweep_points: List[Tensor],
        data_samples: SampleList
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor], Dict[
            str, Tensor]]:
        voxel_dict = dict()
        prev_voxel_dict = dict()
        next_voxel_dict = dict()
        sweep_voxel_dict = dict()

        voxels = []
        coors = []
        point2voxel_maps = []
        voxel_inds = []

        prev_voxels = []
        prev_coors = []
        prev_point2voxel_maps = []
        prev_voxel_inds = []

        next_voxels = []
        next_coors = []
        next_point2voxel_maps = []
        next_voxel_inds = []

        sweep_voxels = []
        sweep_coors = []
        sweep_point2voxel_maps = []
        sweep_voxel_inds = []

        voxel_size = points[0].new_tensor(self.voxel_size)

        for i, (res, prev_res, next_res, sweep_res) in enumerate(
                zip(points, prev_points, next_points, sweep_points)):
            (res_voxels, res_voxel_coors, point2voxel_map,
             inds) = self.voxelize_single(res, voxel_size, i)
            voxels.append(res_voxels)
            coors.append(res_voxel_coors)
            point2voxel_maps.append(point2voxel_map)
            voxel_inds.append(inds)

            (prev_res_voxels, prev_res_voxel_coors, prev_point2voxel_map,
             prev_inds) = self.voxelize_single(prev_res, voxel_size, i)
            prev_voxels.append(prev_res_voxels)
            prev_coors.append(prev_res_voxel_coors)
            prev_point2voxel_maps.append(prev_point2voxel_map)
            prev_voxel_inds.append(prev_inds)

            (next_res_voxels, next_res_voxel_coors, next_point2voxel_map,
             next_inds) = self.voxelize_single(next_res, voxel_size, i)
            next_voxels.append(next_res_voxels)
            next_coors.append(next_res_voxel_coors)
            next_point2voxel_maps.append(next_point2voxel_map)
            next_voxel_inds.append(next_inds)

            (sweep_res_voxels, sweep_res_voxel_coors, sweep_point2voxel_map,
             sweep_inds) = self.voxelize_single(sweep_res, voxel_size, i)
            sweep_voxels.append(sweep_res_voxels)
            sweep_coors.append(sweep_res_voxel_coors)
            sweep_point2voxel_maps.append(sweep_point2voxel_map)
            sweep_voxel_inds.append(sweep_inds)

        voxels = torch.cat(voxels, dim=0)
        coors = torch.cat(coors, dim=0)

        prev_voxels = torch.cat(prev_voxels, dim=0)
        prev_coors = torch.cat(prev_coors, dim=0)

        next_voxels = torch.cat(next_voxels, dim=0)
        next_coors = torch.cat(next_coors, dim=0)

        sweep_voxels = torch.cat(sweep_voxels, dim=0)
        sweep_coors = torch.cat(sweep_coors, dim=0)

        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors
        voxel_dict['point2voxel_maps'] = point2voxel_maps
        voxel_dict['voxel_inds'] = voxel_inds

        prev_voxel_dict['voxels'] = prev_voxels
        prev_voxel_dict['coors'] = prev_coors
        prev_voxel_dict['point2voxel_maps'] = prev_point2voxel_maps
        prev_voxel_dict['voxel_inds'] = prev_voxel_inds

        next_voxel_dict['voxels'] = next_voxels
        next_voxel_dict['coors'] = next_coors
        next_voxel_dict['point2voxel_maps'] = next_point2voxel_maps
        next_voxel_dict['voxel_inds'] = next_voxel_inds

        sweep_voxel_dict['voxels'] = sweep_voxels
        sweep_voxel_dict['coors'] = sweep_coors
        sweep_voxel_dict['point2voxel_maps'] = sweep_point2voxel_maps
        sweep_voxel_dict['voxel_inds'] = sweep_voxel_inds

        return voxel_dict, prev_voxel_dict, next_voxel_dict, sweep_voxel_dict

    def ravel_hash(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2, x.shape

        x = x - np.min(x, axis=0)
        x = x.astype(np.uint64, copy=False)
        xmax = np.max(x, axis=0).astype(np.uint64) + 1

        h = np.zeros(x.shape[0], dtype=np.uint64)
        for k in range(x.shape[1] - 1):
            h += x[:, k]
            h *= xmax[k + 1]
        h += x[:, -1]
        return h

    def sparse_quantize(self,
                        coords: np.ndarray,
                        return_index: bool = False,
                        return_inverse: bool = False) -> List[np.ndarray]:
        _, indices, inverse_indices = np.unique(
            self.ravel_hash(coords), return_index=True, return_inverse=True)

        outputs = []
        if return_index:
            outputs += [indices]
        if return_inverse:
            outputs += [inverse_indices]
        return outputs


@MODELS.register_module()
class DownstreamDataPreprocessor(ImgDataPreprocessor):

    def __init__(self,
                 voxel_size: Sequence[float],
                 voxel_type: str = 'cubic',
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 to_rgb: bool = False,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: bool = False) -> None:
        super(DownstreamDataPreprocessor, self).__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking)
        self._channel_conversion = to_rgb or bgr_to_rgb or rgb_to_bgr
        self.voxel_size = voxel_size
        self.voxel_type = voxel_type

    def forward(self, data: dict, training: bool = False) -> dict:
        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']

            voxel_dict = self.voxelize(inputs['points'], data_samples)
            batch_inputs['voxels'] = voxel_dict

        if 'imgs' in inputs:
            imgs = inputs['imgs']

            if data_samples is not None:
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample in data_samples:
                    data_sample.set_metainfo(
                        {'batch_input_shape': batch_input_shape})

            batch_inputs['imgs'] = imgs

        return {'inputs': batch_inputs, 'data_samples': data_samples}

    def preprocess_img(self, _batch_img: Tensor) -> Tensor:
        if self._channel_conversion:
            _batch_img = _batch_img[[2, 1, 0], ...]
        _batch_img = _batch_img.float()
        if self._enable_normalize:
            if self.mean.shape[0] == 3:
                assert _batch_img.dim() == 3 and _batch_img.shape[0] == 3
            _batch_img = (_batch_img - self.mean) / self.std
        return _batch_img

    def collate_data(self, data: dict) -> dict:
        data = self.cast_data(data)

        if 'imgs' in data['inputs']:
            _batch_imgs = data['inputs']['imgs']

            assert is_seq_of(_batch_imgs, Tensor)

            batch_imgs = []

            for _batch_img in _batch_imgs:
                _batch_img = [self.preprocess_img(_img) for _img in _batch_img]
                _batch_img = torch.stack(_batch_img, dim=0)
                batch_imgs.append(_batch_img)

            batch_imgs = torch.concat(batch_imgs, dim=0)

            data['inputs']['imgs'] = batch_imgs

        data.setdefault('data_samples', None)

        return data

    @torch.no_grad()
    def voxelize(self, points: List[Tensor],
                 data_samples: SampleList) -> Dict[str, Tensor]:
        voxel_dict = dict()

        voxels = []
        coors = []
        point2voxel_maps = []
        voxel_inds = []

        voxel_size = points[0].new_tensor(self.voxel_size)

        for i, res in enumerate(points):
            if self.voxel_type == 'cubic':
                res_coors = torch.round(res[:, :3] / voxel_size).int()
            elif self.voxel_type == 'cylinder':
                rho = torch.sqrt(res[:, 0]**2 + res[:, 1]**2)
                phi = torch.atan2(res[:, 1], res[:, 0]) * 180 / np.pi
                polar_res = torch.stack((rho, phi, res[:, 2]), dim=1)
                res_coors = torch.round(polar_res[:, :3] / voxel_size).int()

            res_coors -= res_coors.min(0)[0]

            res_coors_numpy = res_coors.cpu().numpy()
            inds, point2voxel_map = self.sparse_quantize(
                res_coors_numpy, return_index=True, return_inverse=True)
            point2voxel_map = torch.from_numpy(point2voxel_map).cuda()
            inds = torch.from_numpy(inds).cuda()
            res_voxel_coors = res_coors[inds]
            res_voxels = res[inds]
            res_voxel_coors = F.pad(
                res_voxel_coors, (0, 1), mode='constant', value=i)
            voxels.append(res_voxels)
            coors.append(res_voxel_coors)
            point2voxel_maps.append(point2voxel_map)
            voxel_inds.append(inds)

        voxels = torch.cat(voxels, dim=0)
        coors = torch.cat(coors, dim=0)

        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors
        voxel_dict['point2voxel_maps'] = point2voxel_maps
        voxel_dict['voxel_inds'] = voxel_inds

        return voxel_dict

    def ravel_hash(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2, x.shape

        x = x - np.min(x, axis=0)
        x = x.astype(np.uint64, copy=False)
        xmax = np.max(x, axis=0).astype(np.uint64) + 1

        h = np.zeros(x.shape[0], dtype=np.uint64)
        for k in range(x.shape[1] - 1):
            h += x[:, k]
            h *= xmax[k + 1]
        h += x[:, -1]
        return h

    def sparse_quantize(self,
                        coords: np.ndarray,
                        return_index: bool = False,
                        return_inverse: bool = False) -> List[np.ndarray]:
        _, indices, inverse_indices = np.unique(
            self.ravel_hash(coords), return_index=True, return_inverse=True)

        outputs = []
        if return_index:
            outputs += [indices]
        if return_inverse:
            outputs += [inverse_indices]
        return outputs
