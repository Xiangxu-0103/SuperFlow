from typing import Dict

from mmdet3d.models import EncoderDecoder3D
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from torch import Tensor


@MODELS.register_module()
class VoxelSegmentor(EncoderDecoder3D):

    def __init__(self, freeze_backbone: bool = False, **kwargs) -> None:
        super(VoxelSegmentor, self).__init__(**kwargs)

        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

    def extract_feat(self, batch_inputs_dict: dict) -> dict:
        feat_dict = batch_inputs_dict['voxels'].copy()
        feat_dict = self.backbone(feat_dict)
        if self.with_neck:
            feat_dict = self.neck(feat_dict)
        return feat_dict

    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        feat_dict = self.extract_feat(batch_inputs_dict)
        losses = dict()
        loss_decode = self._decode_head_forward_train(feat_dict,
                                                      batch_data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                feat_dict, batch_data_samples)
            losses.update(loss_aux)
        return losses

    def predict(self,
                batch_inputs_dict: dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)

        feat_dict = self.extract_feat(batch_inputs_dict)
        seg_logits_list = self.decode_head.predict(feat_dict,
                                                   batch_input_metas,
                                                   self.test_cfg)
        for i in range(len(seg_logits_list)):
            seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)

        return self.postprocess_result(seg_logits_list, batch_data_samples)

    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None) -> dict:
        feat_dict = self.extract_feat(batch_inputs_dict)
        return self.decode_head.forward(feat_dict)
