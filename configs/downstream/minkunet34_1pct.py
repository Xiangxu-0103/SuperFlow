_base_ = [
    '../_base_/datasets/nuscenes_seg.py', '../_base_/models/minkunet.py',
    '../_base_/schedules/downstream.py', '../_base_/default_runtime.py'
]

train_dataloader = dict(
    dataset=dict(ann_file='downstream_nuscenes_infos_train_1.pkl'))

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='work_dirs/superflow_vit-b_minkunet34/epoch_100.pth',
            prefix='backbone_3d.')))

default_hooks = dict(checkpoint=dict(save_best='miou'))

work_dir = './work_dirs/superflow_vit-b_minkunet34/minkunet34_1pct'
