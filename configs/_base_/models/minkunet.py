model = dict(
    type='VoxelSegmentor',
    data_preprocessor=dict(
        type='DownstreamDataPreprocessor',
        voxel_size=[0.1, 1, 0.1],
        voxel_type='cylinder'),
    backbone=dict(
        type='MinkUNetBackbone',
        in_channels=4,
        base_channels=32,
        layers=[2, 3, 4, 6, 2, 2, 2, 2],
        planes=[32, 64, 128, 256, 256, 128, 96, 96],
        block_type='basic',
        bn_momentum=0.02),
    decode_head=dict(
        type='LinearHead',
        channels=96,
        num_classes=17,
        dropout_ratio=0,
        ignore_index=16,
        loss_decode=dict(type='mmdet.CrossEntropyLoss', avg_non_ignore=True),
        loss_lovasz=dict(type='LovaszLoss', loss_weight=2.0,
                         reduction='none')),
    train_cfg=dict(),
    test_cfg=dict())
