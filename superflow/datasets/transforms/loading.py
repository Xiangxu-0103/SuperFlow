import copy
import os.path as osp
from typing import List, Optional, Sequence, Union

import mmengine
import mmengine.fileio as fileio
import numpy as np
import torch
from mmcv.transforms import BaseTransform, Compose
from mmdet3d.registry import TRANSFORMS
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from PIL import Image


@TRANSFORMS.register_module()
class LoadMultiSweepsPoints(BaseTransform):

    def __init__(self,
                 sweep_path: str,
                 sweeps_num: int,
                 load_dim: int = 4,
                 use_dim: Union[int, List[int]] = 4,
                 backend_args: Optional[dict] = None) -> None:
        self.sweep_path = sweep_path
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim
        self.use_dim = use_dim
        self.backend_args = backend_args

    def _load_points(self, pts_filename: str) -> np.ndarray:
        try:
            pts_bytes = fileio.get(
                pts_filename, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def transform(self, results: dict) -> dict:
        points = results['points']
        sweep_points_list = [points]
        if 'lidar_sweeps' in results:
            if len(results['lidar_sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['lidar_sweeps']))
            else:
                choices = np.arange(self.sweeps_num)
            for idx in choices:
                sweep = results['lidar_sweeps'][idx]
                points_sweep = self._load_points(
                    osp.join(self.sweep_path,
                             sweep['lidar_points']['lidar_path']))
                points_sweep = points_sweep.reshape(-1, self.load_dim)
                points_sweep = points_sweep[:, self.use_dim]
                lidar2sensor = np.array(sweep['lidar_points']['lidar2sensor'])
                points_sweep[:, :
                             3] = points_sweep[:, :3] @ lidar2sensor[:3, :3]
                points_sweep[:, :3] -= lidar2sensor[:3, 3]
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        sweep_points = points.cat(sweep_points_list)
        results['sweep_points'] = sweep_points
        return results


@TRANSFORMS.register_module()
class LoadMultiModalityData(BaseTransform):

    def __init__(self,
                 superpixel_root: str,
                 num_cameras: int = 6,
                 min_dist: float = 1.0) -> None:
        self.superpixel_root = superpixel_root
        self.min_dist = min_dist
        self.num_cameras = num_cameras

    def transform(self, results: dict) -> dict:
        points = results['points'].numpy()
        pc_original = LidarPointCloud(points.T)
        pairing_points = np.empty(0, dtype=np.int64)
        pairing_images = np.empty((0, 3), dtype=np.int64)

        if 'sweep_points' in results:
            sweep_points = results['sweep_points'].numpy()
            sweep_pc_original = LidarPointCloud(sweep_points.T)
            sweep_pairing_points = np.empty(0, dtype=np.int64)
            sweep_pairing_images = np.empty((0, 3), dtype=np.int64)

        images = []
        superpixels = []

        if 'camera_list' in results:
            camera_list = results['camera_list']
        else:
            camera_list = [
                'CAM_FRONT',
                'CAM_FRONT_LEFT',
                'CAM_FRONT_RIGHT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT',
            ]
            camera_list = np.random.choice(
                camera_list, size=self.num_cameras, replace=False)
            np.random.shuffle(camera_list)
            results['camera_list'] = camera_list

        for i, cam in enumerate(camera_list):
            # load point clouds
            pc = copy.deepcopy(pc_original)

            # load camera images
            img = np.array(Image.open(results['images'][cam]['img_path']))

            # load superpixels
            sp_path = osp.join(
                self.superpixel_root,
                results['images'][cam]['sample_data_token'] + '.png')
            sp = np.array(Image.open(sp_path))

            # transform the point cloud to the ego vehicle frame for the
            # timestamp of the sweep.
            pc.rotate(results['lidar2ego_rotation'])
            pc.translate(results['lidar2ego_translation'])

            # transform from ego to the global frame.
            pc.rotate(results['ego2global_rotation'])
            pc.translate(results['ego2global_translation'])

            # transform from global frame to the ego vehicle frame for the
            # timestamp of the image.
            pc.translate(-results['images'][cam]['ego2global_translation'])
            pc.rotate(results['images'][cam]['ego2global_rotation'].T)

            # transform from ego to the camera.
            pc.translate(-results['images'][cam]['sensor2ego_translation'])
            pc.rotate(results['images'][cam]['sensor2ego_rotation'].T)

            # camera frame z axis points away from the camera
            depths = pc.points[2, :]

            # matrix multiplication with camera-matrix + renormalization.
            points = view_points(
                pc.points[:3, :],
                results['images'][cam]['cam_intrinsic'],
                normalize=True)

            # Remove points that are either outside or behind the camera.
            # Also make sure points are at least 1m in front of the camera to
            # avoid seeing the lidar points on the camera.
            points = points[:2].T
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > self.min_dist)
            mask = np.logical_and(mask, points[:, 0] > 0)
            mask = np.logical_and(mask, points[:, 0] < img.shape[1] - 1)
            mask = np.logical_and(mask, points[:, 1] > 0)
            mask = np.logical_and(mask, points[:, 1] < img.shape[0] - 1)
            matching_points = np.where(mask)[0]
            matching_pixels = np.round(
                np.flip(points[matching_points], axis=1)).astype(np.int64)
            images.append(img / 255.)
            superpixels.append(sp)
            pairing_points = np.concatenate((pairing_points, matching_points))
            pairing_images = np.concatenate(
                (pairing_images,
                 np.concatenate((np.ones(
                     (matching_pixels.shape[0], 1), dtype=np.int64) * i,
                                 matching_pixels),
                                axis=1)))

            if 'sweep_points' in results:
                # load point clouds
                sweep_pc = copy.deepcopy(sweep_pc_original)

                # transform the point cloud to the ego vehicle frame for the
                # timestamp of the sweep.
                sweep_pc.rotate(results['lidar2ego_rotation'])
                sweep_pc.translate(results['lidar2ego_translation'])

                # transform from ego to the global frame.
                sweep_pc.rotate(results['ego2global_rotation'])
                sweep_pc.translate(results['ego2global_translation'])

                # transform from global frame to the ego vehicle frame for the
                # timestamp of the image.
                sweep_pc.translate(
                    -results['images'][cam]['ego2global_translation'])
                sweep_pc.rotate(
                    results['images'][cam]['ego2global_rotation'].T)

                # transform from ego to the camera.
                sweep_pc.translate(
                    -results['images'][cam]['sensor2ego_translation'])
                sweep_pc.rotate(
                    results['images'][cam]['sensor2ego_rotation'].T)

                # camera frame z axis points away from the camera
                sweep_depths = sweep_pc.points[2, :]

                # matrix multiplication with camera-matrix + renormalization.
                sweep_points = view_points(
                    sweep_pc.points[:3, :],
                    results['images'][cam]['cam_intrinsic'],
                    normalize=True)

                # Remove points that are either outside or behind the camera.
                # Also make sure points are at least 1m in front of the camera
                # to avoid seeing the lidar points on the camera.
                sweep_points = sweep_points[:2].T
                sweep_mask = np.ones(sweep_depths.shape[0], dtype=bool)
                sweep_mask = np.logical_and(sweep_mask,
                                            sweep_depths > self.min_dist)
                sweep_mask = np.logical_and(sweep_mask, sweep_points[:, 0] > 0)
                sweep_mask = np.logical_and(
                    sweep_mask, sweep_points[:, 0] < img.shape[1] - 1)
                sweep_mask = np.logical_and(sweep_mask, sweep_points[:, 1] > 0)
                sweep_mask = np.logical_and(
                    sweep_mask, sweep_points[:, 1] < img.shape[0] - 1)
                sweep_matching_points = np.where(sweep_mask)[0]
                sweep_matching_pixels = np.round(
                    np.flip(sweep_points[sweep_matching_points],
                            axis=1)).astype(np.int64)
                sweep_pairing_points = np.concatenate(
                    (sweep_pairing_points, sweep_matching_points))
                sweep_pairing_images = np.concatenate(
                    (sweep_pairing_images,
                     np.concatenate((np.ones(
                         (sweep_matching_pixels.shape[0], 1), dtype=np.int64) *
                                     i, sweep_matching_pixels),
                                    axis=1)))

        results['imgs'] = torch.tensor(
            np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))
        results['superpixels'] = torch.tensor(np.stack(superpixels))
        results['pairing_points'] = pairing_points
        results['pairing_images'] = pairing_images
        if 'sweep_points' in results:
            results['sweep_pairing_points'] = sweep_pairing_points
            results['sweep_pairing_images'] = sweep_pairing_images
        return results


@TRANSFORMS.register_module()
class LoadMultiFrameDataset(BaseTransform):

    def __init__(self, frame: int, transforms: Sequence[dict]) -> None:
        self.frame = frame
        self.transforms = Compose(transforms)

    def transform(self, results: dict) -> dict:
        assert 'dataset' in results

        dataset = results['dataset']

        # load pre_frame
        if results[f'have_prev_{self.frame}_frame']:
            index = results['sample_idx'] - self.frame
        else:
            index = results['sample_idx']

        prev_results = dataset.get_data_info(index)
        prev_results['camera_list'] = results['camera_list']
        prev_results = self.transforms(prev_results)

        results['prev_points'] = prev_results['points']
        results['prev_imgs'] = prev_results['imgs']
        results['prev_pairing_points'] = prev_results['pairing_points']
        results['prev_pairing_images'] = prev_results['pairing_images']
        results['prev_superpixels'] = prev_results['superpixels']

        # load next_frame
        if results[f'have_next_{self.frame}_frame']:
            index = results['sample_idx'] + self.frame
        else:
            index = results['sample_idx']

        next_results = dataset.get_data_info(index)
        next_results['camera_list'] = results['camera_list']
        next_results = self.transforms(next_results)

        results['next_points'] = next_results['points']
        results['next_imgs'] = next_results['imgs']
        results['next_pairing_points'] = next_results['pairing_points']
        results['next_pairing_images'] = next_results['pairing_images']
        results['next_superpixels'] = next_results['superpixels']
        return results
