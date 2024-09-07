_base_ = [
    '../_base_/datasets/nuscenes_pretrain.py', '../_base_/models/superflow.py',
    '../_base_/schedules/pretrain.py', '../_base_/default_runtime.py'
]

work_dir = './work_dirs/superflow_vit-b_minkunet34'
