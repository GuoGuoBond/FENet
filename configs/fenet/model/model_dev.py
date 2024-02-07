model = dict(
    type='FE_Net',
    pts_backbone=dict(
        type='PointNet2SAMSG',
        in_channels=4,
        num_points=(8192, 1024, 512),
        radii=((0.1, 0.5), (0.5, 1.0), (1.0, 2.0)),
        num_samples=((16, 32), (16, 32), (16, 32)),
        sa_channels=(((16, 16, 32), (32, 32, 64)),
                     ((64, 64, 128), (64, 96, 128)),
                     ((128, 128, 256), (128, 256, 256))),
        aggregation_channels=(64, 128, 256),
        confidence_mlps=(0, 0, 128),
        num_classes=3,
        fps_mods=(('D-FPS'), ('D-FPS'), ('CS')),
        fps_sample_range_lists=((-1), (-1), (-1)),
        norm_cfg=dict(type='BN2d', eps=1e-5, momentum=0.1),
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False,
            cylinder_group=True),
        ),


    pts_bbox_head=dict(
        type='FusionHead',
        in_channels=256,
        confidence_mlps=(256, ),
        num_classes=3,
        transformer_cfg=dict(
            d_model=256, nhead=4, dropout=0.1, embedding_num_per_cls=30,
            mu=0.999, num_classes=3, prototype_num_per_class=30
        ),
        vote_module_cfg=dict(
            in_channels=256,
            num_points=256,
            gt_per_seed=1,
            conv_channels=(128, ),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1),
            with_res_feat=False,
            vote_xyz_range=(3.0, 3.0, 2.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModuleMSG',
            num_point=512,
            radii=(2.0, 4.0),
            sample_nums=(16, 32),
            mlp_channels=((256, 256, 256, 512), (256, 256, 512, 1024)),
            norm_cfg=dict(type='BN2d', eps=1e-5, momentum=0.1),
            use_xyz=True,
            normalize_xyz=False,
            bias=False),
        pred_layer_cfg=dict(
            in_channels=1536,
            shared_conv_channels=(512, ),
            cls_conv_channels=(256, 256),
            reg_conv_channels=(256, 256),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1),
            bias=False),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1),
        objectness_loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        center_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        dir_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        corner_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        vote_loss=dict(type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        num_cls_layer=2,
        base_cls_loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        bbox_coder=dict(
            type='AnchorFreeBBoxCoder', num_dir_bins=12, with_rot=True)),

    train_cfg=dict(
        pts=dict(sample_mod='spec',
                 pos_distance_thr=10.0,
                 expand_dims_length=0.05,
                 stage='train')),
    test_cfg=dict(
        pts=dict(nms_cfg=dict(type='nms', iou_thr=0.1),
        sample_mod='spec',
        score_thr=0.0,
        per_class_proposal=False,
        max_output_num=100)),
    pretrained=dict(
        img=None,
        pts=None
    ))




