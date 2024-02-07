# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
from torch import nn as nn

from mmdet3d.ops import build_sa_module
from ..builder import BACKBONES
from .base_pointnet import BasePointNet
from mmdet3d.models.model_utils import Confidence_mlps


def shared_mlp(in_channel, confidence_mlp_channel, num_classes):
    # input (B, C, N)
    shared_mlp = []
    shared_mlp.extend([
        nn.Conv1d(in_channel, confidence_mlp_channel, kernel_size=1, bias=False),
        nn.BatchNorm1d(confidence_mlp_channel),
        nn.ReLU(),
        nn.Conv1d(confidence_mlp_channel, num_classes, kernel_size=1, bias=True)
    ])
    return shared_mlp

def get_cls_index(cls_features, num_points): # (B, N, Class)
    cls_features_max, class_pred = cls_features.max(dim=-1)  # max之后的值， max值在原数据dim中的索引
    score_pred = torch.sigmoid(cls_features_max)  # B,N
    score_picked, sample_idx = torch.topk(score_pred, num_points, dim=-1)  # topk==npoint，返回保留的值，索引
    sample_idx = sample_idx.int()
    return sample_idx

@BACKBONES.register_module()
class PointNet2SAMSG(BasePointNet):
    """PointNet2 with Multi-scale grouping.

    Args:
        in_channels (int): Input channels of point cloud.
        num_points (tuple[int]): The number of points which each SA
            module samples.
        radii (tuple[float]): Sampling radii of each SA module.
        num_samples (tuple[int]): The number of samples for ball
            query in each SA module.
        sa_channels (tuple[tuple[int]]): Out channels of each mlp in SA module.
        aggregation_channels (tuple[int]): Out channels of aggregation
            multi-scale grouping features.
        fps_mods (tuple[int]): Mod of FPS for each SA module.
        fps_sample_range_lists (tuple[tuple[int]]): The number of sampling
            points which each SA module samples.
        dilated_group (tuple[bool]): Whether to use dilated ball query for
        out_indices (Sequence[int]): Output from which stages.
        norm_cfg (dict): Config of normalization layer.
        sa_cfg (dict): Config of set abstraction module, which may contain
            the following keys and values:

            - pool_mod (str): Pool method ('max' or 'avg') for SA modules.
            - use_xyz (bool): Whether to use xyz as a part of features.
            - normalize_xyz (bool): Whether to normalize xyz with radii in
              each SA module.
    """

    def __init__(self,
                 in_channels,
                 num_points=(2048, 1024, 512, 256),
                 radii=((0.2, 0.4, 0.8), (0.4, 0.8, 1.6), (1.6, 3.2, 4.8)),
                 num_samples=((32, 32, 64), (32, 32, 64), (32, 32, 32)),
                 sa_channels=(((16, 16, 32), (16, 16, 32), (32, 32, 64)),
                              ((64, 64, 128), (64, 64, 128), (64, 96, 128)),
                              ((128, 128, 256), (128, 192, 256), (128, 256,
                                                                  256))),
                 aggregation_channels=(64, 128, 256),
                 confidence_mlps=(0, 256, 512), # 根据类别下采样
                 num_classes=3,
                 fps_mods=(('D-FPS'), ('FS'), ('F-FPS', 'D-FPS')),
                 fps_sample_range_lists=((-1), (-1), (512, -1)),
                 dilated_group=(True, True, True),
                 out_indices=(2, ),
                 norm_cfg=dict(type='BN2d'),
                 sa_cfg=dict(
                     type='PointSAModuleMSG',
                     pool_mod='max',
                     use_xyz=True,
                     normalize_xyz=False),
                 cylinder_group_sa_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_sa = len(sa_channels)
        self.out_indices = out_indices
        assert max(out_indices) < self.num_sa
        assert len(num_points) == len(radii) == len(num_samples) == len(
            sa_channels)
        self.num_points = num_points
        if aggregation_channels is not None:
            assert len(sa_channels) == len(aggregation_channels)
        else:
            aggregation_channels = [None] * len(sa_channels)

        self.SA_modules = nn.ModuleList()
        self.aggregation_mlps = nn.ModuleList()
        self.confidence_mlps = nn.ModuleList()
        sa_in_channel = in_channels - 3  # number of channels without xyz
        skip_channel_list = [sa_in_channel]
        self.fps_mods = fps_mods
        for sa_index in range(self.num_sa):
            cur_sa_mlps = list(sa_channels[sa_index])
            sa_out_channel = 0
            for radius_index in range(len(radii[sa_index])):
                cur_sa_mlps[radius_index] = [sa_in_channel] + list(
                    cur_sa_mlps[radius_index])
                sa_out_channel += cur_sa_mlps[radius_index][-1]

            if isinstance(fps_mods[sa_index], tuple):
                cur_fps_mod = list(fps_mods[sa_index])
            else:
                cur_fps_mod = list([fps_mods[sa_index]])

            if isinstance(fps_sample_range_lists[sa_index], tuple):
                cur_fps_sample_range_list = list(
                    fps_sample_range_lists[sa_index])
            else:
                cur_fps_sample_range_list = list(
                    [fps_sample_range_lists[sa_index]])

            # SA层
            self.SA_modules.append(
                build_sa_module(
                    num_point=num_points[sa_index],
                    radii=radii[sa_index],
                    sample_nums=num_samples[sa_index],
                    mlp_channels=cur_sa_mlps,
                    fps_mod=cur_fps_mod,
                    fps_sample_range_list=cur_fps_sample_range_list,
                    dilated_group=dilated_group[sa_index],
                    norm_cfg=norm_cfg,
                    cfg=sa_cfg,
                    bias=False))
            skip_channel_list.append(sa_out_channel)

            # 构造分类层
            cur_confidence_mlp = confidence_mlps[sa_index]
            if cur_confidence_mlp == 0:
                self.confidence_mlps.append(None)
            else:
                self.confidence_mlps.append(
                    Confidence_mlps(sa_in_channel, cur_confidence_mlp, num_classes))

            # 聚合层 聚合sa输出的多维度特征
            cur_aggregation_channel = aggregation_channels[sa_index]
            if cur_aggregation_channel is None:
                self.aggregation_mlps.append(None)
                sa_in_channel = sa_out_channel
            else:
                self.aggregation_mlps.append(
                    ConvModule(
                        sa_out_channel,
                        cur_aggregation_channel,
                        conv_cfg=dict(type='Conv1d'),
                        norm_cfg=dict(type='BN1d'),
                        kernel_size=1,
                        bias=False))
                sa_in_channel = cur_aggregation_channel

        self.cylinder_group_sa_cfg = cylinder_group_sa_cfg
        if cylinder_group_sa_cfg != None:
            self.cylinder_SA_modules = nn.ModuleList()
            self.cylinder_aggregation_mlps = nn.ModuleList()
            self.fusion_mlps = nn.ModuleList()
            c_cfg = cylinder_group_sa_cfg
            # cylinder_SA层
            self.cylinder_SA_modules.append(
                build_sa_module(
                    num_point=c_cfg.num_points,
                    radii=c_cfg.radii,
                    sample_nums=c_cfg.num_samples,
                    mlp_channels=c_cfg.sa_channels,
                    norm_cfg=c_cfg.norm_cfg,
                    cfg=c_cfg.sa_cfg,
                    bias=False))

            # cylinder_aggregation层
            cylinder_sa_out_channel = 0
            for i in range(len(c_cfg.sa_channels)):
                cylinder_sa_out_channel = cylinder_sa_out_channel + c_cfg.sa_channels[i][-1]

            self.cylinder_aggregation_mlps.append(
                ConvModule(
                    cylinder_sa_out_channel,
                    c_cfg.cylinder_aggregation_channels,
                    conv_cfg=dict(type='Conv1d'),
                    norm_cfg=dict(type='BN1d'),
                    kernel_size=1,
                    bias=False))

            # fusion cylinder_features 和 原始_features 层
            self.fusion_mlps.append(
                ConvModule(
                    c_cfg.cylinder_aggregation_channels+cur_aggregation_channel,
                    cur_aggregation_channel,
                    conv_cfg=dict(type='Conv1d'),
                    norm_cfg=dict(type='BN1d'),
                    kernel_size=1,
                    bias=False))

    @auto_fp16(apply_to=('points', ))
    def forward(self, points):
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            dict[str, torch.Tensor]: Outputs of the last SA module.

                - sa_xyz (torch.Tensor): The coordinates of sa features.
                - sa_features (torch.Tensor): The features from the
                    last Set Aggregation Layers.
                - sa_indices (torch.Tensor): Indices of the
                    input points.
        """
        xyz, features = self._split_point_feats(points)

        batch, num_points = xyz.shape[:2]
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(
            batch, 1).long()

        sa_xyz = [xyz]  # List [(B, N, xyz),]
        sa_features = [features]  # List [(B, C, N),]
        sa_indices = [indices]  # List [(B, N),]

        cls_xyz = []
        cls_features = []
        cls_indices = []

        for i in range(self.num_sa):

            # 是否使用基于类别的下采样 fixme 加距离
            if self.fps_mods[i] == 'CS':
                cur_cls_features, _ = self.confidence_mlps[i](sa_features[i]) #(B, Class, N)
                _, _, sample_idx = self.confidence_mlps[i].get_topk(
                    sa_xyz[i], sa_features[i], sa_indices[i], self.num_points[i])
                # (B, Npoints)
                cur_xyz, cur_features, cur_indices = self.SA_modules[i](
                    sa_xyz[i], sa_features[i], indices=sample_idx)
                cls_xyz.append(sa_xyz[i])
                cls_features.append(cur_cls_features)
                cls_indices.append(sa_indices[i])
            else:
                cur_xyz, cur_features, cur_indices = self.SA_modules[i](
                    sa_xyz[i], sa_features[i])

            if self.aggregation_mlps[i] is not None:
                cur_features = self.aggregation_mlps[i](cur_features)

            sa_xyz.append(cur_xyz)
            sa_features.append(cur_features)
            sa_indices.append(
                torch.gather(sa_indices[-1], 1, cur_indices.long()))  # 找到当前索引在原索引中的位置

        if self.cylinder_group_sa_cfg != None:
            _, cylinder_sa_features, _ = \
                self.cylinder_SA_modules[0](sa_xyz[0], sa_features[0], indices=sample_idx)
            cylinder_sa_features = self.cylinder_aggregation_mlps[0](cylinder_sa_features)
            sa_features[-1] = self.fusion_mlps[0](torch.cat([cur_features, cylinder_sa_features], dim=1))



        return dict(
            sa_xyz=sa_xyz,
            sa_features=sa_features,
            sa_indices=sa_indices,
            cls_xyz=cls_xyz,
            cls_features=cls_features,
            cls_indices=cls_indices)
