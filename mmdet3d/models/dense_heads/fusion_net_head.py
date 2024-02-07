# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core.bbox.structures import (DepthInstance3DBoxes,
                                          LiDARInstance3DBoxes,
                                          rotation_3d_in_axis)
from mmdet.core import multi_apply
from ..builder import HEADS, build_loss
from .vote_head import VoteHead
from mmcv.ops import PointsSampler as Points_Sampler

@HEADS.register_module()
class FusionHead(VoteHead):
    r"""Bbox head of `3DSSD <https://arxiv.org/abs/2002.10187>`_.

    Args:
        num_classes (int): The number of class.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for encoding and
            decoding boxes.
        in_channels (int): The number of input feature channel.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
        vote_module_cfg (dict): Config of VoteModule for point-wise votes.
        vote_aggregation_cfg (dict): Config of vote aggregation layer.
        pred_layer_cfg (dict): Config of classfication and regression
            prediction layers.
        conv_cfg (dict): Config of convolution in prediction layer.
        norm_cfg (dict): Config of BN in prediction layer.
        act_cfg (dict): Config of activation in prediction layer.
        objectness_loss (dict): Config of objectness loss.
        center_loss (dict): Config of center loss.
        dir_class_loss (dict): Config of direction classification loss.
        dir_res_loss (dict): Config of direction residual regression loss.
        size_res_loss (dict): Config of size residual regression loss.
        corner_loss (dict): Config of bbox corners regression loss.
        vote_loss (dict): Config of candidate points regression loss.
    """

    def __init__(self,
                 num_classes,
                 bbox_coder,
                 in_channels=256,
                 train_cfg=None,
                 test_cfg=None,
                 vote_module_cfg=None,
                 vote_aggregation_cfg=None,
                 pred_layer_cfg=None,
                 objectness_loss=None,
                 center_loss=None,
                 dir_class_loss=None,
                 dir_res_loss=None,
                 size_res_loss=None,
                 corner_loss=None,
                 vote_loss=None,
                 init_cfg=None,
                 base_cls_loss=None,
                 num_cls_layer=0,
                 transformer_cfg=None):
        super(FusionHead, self).__init__(
            num_classes,
            bbox_coder,
            in_channels=in_channels,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            vote_module_cfg=vote_module_cfg,
            vote_aggregation_cfg=vote_aggregation_cfg,
            pred_layer_cfg=pred_layer_cfg,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            objectness_loss=objectness_loss,
            center_loss=center_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_class_loss=None,
            size_res_loss=size_res_loss,
            semantic_loss=None,
            init_cfg=init_cfg,
            confidence_mlps=confidence_mlps,
            transformer_cfg=transformer_cfg)

        self.corner_loss = build_loss(corner_loss)
        self.vote_loss = build_loss(vote_loss)
        self.num_candidates = vote_module_cfg['num_points']

        self.num_cls_layer = num_cls_layer
        self.cls_loss = []
        for i in range(num_cls_layer):
            self.cls_loss.append(build_loss(base_cls_loss))

    def _get_cls_out_channels(self):
        """Return the channel number of classification outputs."""
        # Class numbers (k) + objectness (1)
        return self.num_classes

    def _get_reg_out_channels(self):
        """Return the channel number of regression outputs."""
        # Bbox classification and regression
        # (center residual (3), size regression (3)
        # heading class+residual (num_dir_bins*2)),
        return 3 + 3 + self.num_dir_bins * 2

    def _extract_input(self, feat_dict):
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
            torch.Tensor: Indices of input points.
        """
        seed_points = feat_dict['sa_xyz'][-1]
        seed_features = feat_dict['sa_features'][-1]
        seed_indices = feat_dict['sa_indices'][-1]
        cls_points = feat_dict['cls_xyz']
        cls_features = feat_dict['cls_features']
        cls_indices = feat_dict['cls_indices']
        input_points = feat_dict['sa_xyz'][0]
        input_features = feat_dict['sa_features'][0]
        input_indices = feat_dict['sa_indices'][0]


        return seed_points, seed_features, seed_indices,\
            cls_points, cls_features, cls_indices, \
            input_points, input_features, input_indices

    @force_fp32(apply_to=('bbox_preds', ))
    def loss(self,
             bbox_preds,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             pts_semantic_mask=None,
             pts_instance_mask=None,
             img_metas=None,
             gt_bboxes_ignore=None):
        """Compute loss.

        Args:
            bbox_preds (dict): Predictions from forward of SSD3DHead.
            points (list[torch.Tensor]): Input points.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                bboxes of each sample.
            gt_labels_3d (list[torch.Tensor]): Labels of each sample.
            pts_semantic_mask (list[torch.Tensor]): Point-wise
                semantic mask.
            pts_instance_mask (list[torch.Tensor]): Point-wise
                instance mask.
            img_metas (list[dict]): Contain pcd and img's meta info.
            gt_bboxes_ignore (list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses of 3DSSD.
        """
        targets = self.get_targets(points, gt_bboxes_3d, gt_labels_3d,
                                   pts_semantic_mask, pts_instance_mask,
                                   bbox_preds)
        (vote_targets, center_targets, size_res_targets, dir_class_targets,
         dir_res_targets, mask_targets, centerness_targets, corner3d_targets,
         vote_mask, centerness_weights, box_loss_weights, heading_res_loss_weight,
         cls_targets, cls_weights) = targets

        # calculate centerness loss
        centerness_loss = self.objectness_loss(
            bbox_preds['obj_scores'].transpose(2, 1),
            centerness_targets,
            weight=centerness_weights)

        # calculate center loss
        center_loss = self.center_loss(
            bbox_preds['center_offset'],
            center_targets,
            weight=box_loss_weights.unsqueeze(-1))

        # calculate direction class loss
        dir_class_loss = self.dir_class_loss(
            bbox_preds['dir_class'].transpose(1, 2),
            dir_class_targets,
            weight=box_loss_weights)

        # calculate direction residual loss
        dir_res_loss = self.dir_res_loss(
            bbox_preds['dir_res_norm'],
            dir_res_targets.unsqueeze(-1).repeat(1, 1, self.num_dir_bins),
            weight=heading_res_loss_weight)

        # calculate size residual loss
        size_loss = self.size_res_loss(
            bbox_preds['size'],
            size_res_targets,
            weight=box_loss_weights.unsqueeze(-1))

        # calculate corner loss
        one_hot_dir_class_targets = dir_class_targets.new_zeros(
            bbox_preds['dir_class'].shape)
        one_hot_dir_class_targets.scatter_(2, dir_class_targets.unsqueeze(-1),
                                           1)
        pred_bbox3d = self.bbox_coder.decode(
            dict(
                center=bbox_preds['center'],
                dir_res=bbox_preds['dir_res'],
                dir_class=one_hot_dir_class_targets,
                size=bbox_preds['size']))
        pred_bbox3d = pred_bbox3d.reshape(-1, pred_bbox3d.shape[-1])
        pred_bbox3d = img_metas[0]['box_type_3d'](
            pred_bbox3d.clone(),
            box_dim=pred_bbox3d.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 0.5))
        pred_corners3d = pred_bbox3d.corners.reshape(-1, 8, 3)
        corner_loss = self.corner_loss(
            pred_corners3d,
            corner3d_targets.reshape(-1, 8, 3),
            weight=box_loss_weights.view(-1, 1, 1))

        # calculate vote loss fixme vote_loss
        vote_loss = self.vote_loss(
            bbox_preds['vote_offset'].transpose(1, 2),
            vote_targets,
            weight=vote_mask.unsqueeze(-1))

        # cal cls loss
        cls_loss = []
        for i in range(self.num_cls_layer):
            layer_cls_loss = self.cls_loss[i](
                bbox_preds['cls_features'][i].transpose(1, 2),
                cls_targets[i],
                weight=cls_weights[i]
                )
            cls_loss.append(layer_cls_loss)

        losses = dict(
            centerness_loss=centerness_loss,
            center_loss=center_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_res_loss=size_loss,
            corner_loss=corner_loss,
            vote_loss=vote_loss,
            cls_loss=cls_loss)

        return losses

    def get_targets(self,
                    points,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    pts_semantic_mask=None,
                    pts_instance_mask=None,
                    bbox_preds=None):
        """Generate targets of ssd3d head.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): Labels of each batch.
            pts_semantic_mask (list[torch.Tensor]): Point-wise semantic
                label of each batch.
            pts_instance_mask (list[torch.Tensor]): Point-wise instance
                label of each batch.
            bbox_preds (torch.Tensor): Bounding box predictions of ssd3d head.

        Returns:
            tuple[torch.Tensor]: Targets of ssd3d head.
        """
        # find empty example
        for index in range(len(gt_labels_3d)):
            if len(gt_labels_3d[index]) == 0:
                fake_box = gt_bboxes_3d[index].tensor.new_zeros(
                    1, gt_bboxes_3d[index].tensor.shape[-1])
                gt_bboxes_3d[index] = gt_bboxes_3d[index].new_box(fake_box)
                gt_labels_3d[index] = gt_labels_3d[index].new_zeros(1)

        if pts_semantic_mask is None:
            pts_semantic_mask = [None for i in range(len(gt_labels_3d))]
            pts_instance_mask = [None for i in range(len(gt_labels_3d))]

        aggregated_points = [
            bbox_preds['aggregated_points'][i]
            for i in range(len(gt_labels_3d))
        ]

        # aggregated_indices = [
        #     bbox_preds['aggregated_indices'][i]
        #     for i in range(len(gt_labels_3d))
        # ]

        # fixme seed_points 数目 目前未使用
        seed_points = [
            bbox_preds['seed_points'][i, :self.num_candidates].detach()
            for i in range(len(gt_labels_3d))
        ]

        origin_vote_points = [
            bbox_preds['origin_vote_points'][i]
            for i in range(len(gt_labels_3d))
        ]

        input_points = [
            bbox_preds['input_points'][i]
            for i in range(len(gt_labels_3d))
        ]

        # 把原始数据按batch拆开 List [batch1,batch2,...,]
        cls_points = []
        for batch in range(len(gt_labels_3d)):
            tmp_cls_points = [
                bbox_preds['cls_points'][i][batch].detach()
                for i in range(self.num_cls_layer)
            ]
            cls_points.append(tmp_cls_points)

        cls_indices = []
        for batch in range(len(gt_labels_3d)):
            tmp_cls_indices = [
                bbox_preds['cls_indices'][i][batch].detach()
                for i in range(self.num_cls_layer)
            ]
            cls_indices.append(tmp_cls_indices)

        (vote_targets, center_targets, size_res_targets, dir_class_targets,
         dir_res_targets, mask_targets, centerness_targets, corner3d_targets,
         vote_mask, centerness_positive_mask, centerness_negative_mask,
         cls_targets, cls_pos_mask, cls_neg_mask) = multi_apply(
             self.get_targets_single, points, gt_bboxes_3d, gt_labels_3d,
             pts_semantic_mask, pts_instance_mask, aggregated_points,
             seed_points, cls_points, origin_vote_points, input_points, cls_indices)

        # 将每个batch的结果组合到一起 List: Batch *(N,c) -> (B, N, c)
        center_targets = torch.stack(center_targets)
        dir_class_targets = torch.stack(dir_class_targets)
        dir_res_targets = torch.stack(dir_res_targets)
        size_res_targets = torch.stack(size_res_targets)
        mask_targets = torch.stack(mask_targets)
        centerness_targets = torch.stack(centerness_targets).detach()
        corner3d_targets = torch.stack(corner3d_targets)
        vote_targets = torch.stack(vote_targets)
        centerness_positive_mask = torch.stack(centerness_positive_mask)
        centerness_negative_mask = torch.stack(centerness_negative_mask)
        vote_mask = torch.stack(vote_mask)


        center_targets -= bbox_preds['aggregated_points']


        centerness_weights = (centerness_positive_mask +
                              centerness_negative_mask).unsqueeze(-1).repeat(
                                  1, 1, self.num_classes).float()
        centerness_weights = centerness_weights / \
            (centerness_weights.sum() + 1e-6)

        vote_mask = vote_mask / (vote_mask.sum() + 1e-6)

        box_loss_weights = centerness_positive_mask / (centerness_positive_mask.sum() + 1e-6)

        batch_size, proposal_num = dir_class_targets.shape[:2]
        heading_label_one_hot = dir_class_targets.new_zeros(
            (batch_size, proposal_num, self.num_dir_bins))
        heading_label_one_hot.scatter_(2, dir_class_targets.unsqueeze(-1), 1)
        heading_res_loss_weight = heading_label_one_hot * \
            box_loss_weights.unsqueeze(-1)

        # cls_targets List (every_Batch * every_layer) -> (Batch * every_layer)
        tmp_cls_targets, tmp_cls_pos, tmp_cls_neg = [], [], []
        for layer in range(self.num_cls_layer):
            _cls_target = torch.stack([cls_target[layer] for cls_target in cls_targets])
            tmp_cls_targets.append(_cls_target)

            _cls_pos_mask = torch.stack([_mask[layer] for _mask in cls_pos_mask])
            _cls_neg_mask = torch.stack([_mask[layer] for _mask in cls_neg_mask])
            tmp_cls_pos.append(_cls_pos_mask)
            tmp_cls_neg.append(_cls_neg_mask)
        cls_targets, cls_pos, cls_neg = tmp_cls_targets, tmp_cls_pos, tmp_cls_neg

        cls_weights = []
        for layer in range(self.num_cls_layer):
            weight = (1.0 * cls_pos[layer] + 1.0 * cls_neg[layer]).unsqueeze(-1).repeat(
                                  1, 1, self.num_classes).float()
            weight /= (cls_pos[layer].sum() + 1e-6)
            cls_weights.append(weight)

        return (vote_targets, center_targets, size_res_targets,
                dir_class_targets, dir_res_targets, mask_targets,
                centerness_targets, corner3d_targets, vote_mask,
                centerness_weights, box_loss_weights,
                heading_res_loss_weight, cls_targets, cls_weights)

    # 逐个batch生成所需标签
    def get_targets_single(self,
                           points,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           pts_semantic_mask=None,
                           pts_instance_mask=None,
                           aggregated_points=None,
                           seed_points=None,
                           cls_points=None,
                           origin_vote_points=None,
                           input_points=None,
                           cls_indices=None,
                           aggregated_indices=None):
        """Generate targets of ssd3d head for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth
                boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.
            pts_semantic_mask (torch.Tensor): Point-wise semantic
                label of each batch.
            pts_instance_mask (torch.Tensor): Point-wise instance
                label of each batch.
            aggregated_points (torch.Tensor): Aggregated points from
                candidate points layer.
            seed_points (torch.Tensor): Seed points of candidate points.

        Returns:
            tuple[torch.Tensor]: Targets of ssd3d head.
        """
        assert self.bbox_coder.with_rot or pts_semantic_mask is not None
        gt_bboxes_3d = gt_bboxes_3d.to(points.device)
        valid_gt = gt_labels_3d != -1 # 去除label为-1的gtbox，仅使用三类
        gt_bboxes_3d = gt_bboxes_3d[valid_gt]
        gt_labels_3d = gt_labels_3d[valid_gt]

        # Generate fake GT for empty scene
        if valid_gt.sum() == 0:
            vote_targets = points.new_zeros(self.num_candidates, 3)
            center_targets = points.new_zeros(self.num_candidates, 3)
            size_res_targets = points.new_zeros(self.num_candidates, 3)
            dir_class_targets = points.new_zeros(
                self.num_candidates, dtype=torch.int64)
            dir_res_targets = points.new_zeros(self.num_candidates)
            mask_targets = points.new_zeros(
                self.num_candidates, dtype=torch.int64)
            centerness_targets = points.new_zeros(self.num_candidates,
                                                  self.num_classes)
            corner3d_targets = points.new_zeros(self.num_candidates, 8, 3)
            vote_mask = points.new_zeros(self.num_candidates, dtype=torch.bool)
            positive_mask = points.new_zeros(
                self.num_candidates, dtype=torch.bool)
            negative_mask = points.new_ones(
                self.num_candidates, dtype=torch.bool)
            return (vote_targets, center_targets, size_res_targets,
                    dir_class_targets, dir_res_targets, mask_targets,
                    centerness_targets, corner3d_targets, vote_mask,
                    positive_mask, negative_mask)

        # gtbox的八个角的坐标
        gt_corner3d = gt_bboxes_3d.corners

        # gtbox的几何中心点 长宽高/2  方向类别 方向残差
        (center_targets, size_targets, dir_class_targets,
         dir_res_targets) = self.bbox_coder.encode(gt_bboxes_3d, gt_labels_3d)

        # 点在哪个gtbox内（置1） 每个点对应的gtbox
        points_mask, assignment = self._assign_targets_by_points_inside(
            gt_bboxes_3d, aggregated_points)

        # 每个点对应的lable
        center_targets = center_targets[assignment]
        size_res_targets = size_targets[assignment]
        mask_targets = gt_labels_3d[assignment]
        dir_class_targets = dir_class_targets[assignment]
        dir_res_targets = dir_res_targets[assignment]
        corner3d_targets = gt_corner3d[assignment]

        # Centerness loss targets
        centerness_targets, centerness_positive_mask, centerness_negative_mask = \
        self._get_centreness_soft_target(gt_bboxes_3d, gt_labels_3d, aggregated_points, enlarge=True)

        # centerness_targets, centerness_positive_mask, centerness_negative_mask = \
        #     self._get_closest_k_points_to_gt_boxes_target(
        #         gt_bboxes_3d, gt_labels_3d, input_points, aggregated_indices, enlarge=0.1, topk=256, hard=False)


        # Vote loss targets
        vote_targets, vote_mask = self._get_vote_target(gt_bboxes_3d, origin_vote_points)

        # class_sample targets v1
        # cls_targets, cls_pos_mask, cls_neg_mask = [], [], []
        # for i in range(len(cls_points)):
        #     _cls_target, _cls_pos_mask, _cls_neg_mask = self._get_cls_target(
        #         gt_bboxes_3d, gt_labels_3d, cls_points[i], enlarge=True)
        #     cls_targets.append(_cls_target)
        #     cls_pos_mask.append(_cls_pos_mask)
        #     cls_neg_mask.append(_cls_neg_mask)

        # class_sample targets v2(closest k point)
        cls_targets, cls_pos_mask, cls_neg_mask = [], [], []
        for i in range(len(cls_indices)):
            closest_k_targets, pos_mask, neg_mask = self._get_closest_k_points_to_gt_boxes_target(
                gt_bboxes_3d, gt_labels_3d, input_points, cls_indices[i], enlarge=0.25, topk=256, hard=False)
            cls_targets.append(closest_k_targets)
            cls_pos_mask.append(pos_mask)
            cls_neg_mask.append(neg_mask)

        # class_sample targets v3(Skeleton Point with v1)
        # cls_targets, cls_pos_mask, cls_neg_mask = [], [], []
        # for i in range(len(cls_points)):
        #     _cls_target, _cls_pos_mask, _cls_neg_mask = self._get_sp_to_gt_boxes_target(
        #         gt_bboxes_3d, gt_labels_3d, input_points, cls_points[i], cls_indices[i], enlarge=0.25, topk=256)
        #     cls_targets.append(_cls_target)
        #     cls_pos_mask.append(_cls_pos_mask)
        #     cls_neg_mask.append(_cls_neg_mask)

        return (vote_targets, center_targets, size_res_targets,
                dir_class_targets, dir_res_targets, mask_targets,
                centerness_targets, corner3d_targets, vote_mask,
                centerness_positive_mask, centerness_negative_mask
                , cls_targets, cls_pos_mask, cls_neg_mask)

    def get_bboxes(self, points, bbox_preds, input_metas, rescale=False):
        """Generate bboxes from 3DSSD head predictions.

        Args:
            points (torch.Tensor): Input points.
            bbox_preds (dict): Predictions from sdd3d head.
            input_metas (list[dict]): Point cloud and image's meta info.
            rescale (bool): Whether to rescale bboxes.

        Returns:
            list[tuple[torch.Tensor]]: Bounding boxes, scores and labels.
        """
        # decode boxes
        sem_scores = F.sigmoid(bbox_preds['obj_scores']).transpose(1, 2)
        obj_scores = sem_scores.max(-1)[0]
        bbox3d = self.bbox_coder.decode(bbox_preds)

        batch_size = bbox3d.shape[0]
        results = list()

        for b in range(batch_size):
            bbox_selected, score_selected, labels = self.multiclass_nms_single(
                obj_scores[b], sem_scores[b], bbox3d[b], points[b, ..., :3],
                input_metas[b])

            bbox = input_metas[b]['box_type_3d'](
                bbox_selected.clone(),
                box_dim=bbox_selected.shape[-1],
                with_yaw=self.bbox_coder.with_rot)
            results.append((bbox, score_selected, labels))

        return results

    def multiclass_nms_single(self, obj_scores, sem_scores, bbox, points,
                              input_meta):
        """Multi-class nms in single batch.

        Args:
            obj_scores (torch.Tensor): Objectness score of bounding boxes.
            sem_scores (torch.Tensor): Semantic class score of bounding boxes.
            bbox (torch.Tensor): Predicted bounding boxes.
            points (torch.Tensor): Input points.
            input_meta (dict): Point cloud and image's meta info.

        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """
        bbox = input_meta['box_type_3d'](
            bbox.clone(),
            box_dim=bbox.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 0.5))

        if isinstance(bbox, (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            box_indices = bbox.points_in_boxes_all(points)
            nonempty_box_mask = box_indices.T.sum(1) >= 0
        else:
            raise NotImplementedError('Unsupported bbox type!')

        corner3d = bbox.corners
        minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
        minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
        minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]

        bbox_classes = torch.argmax(sem_scores, -1)
        nms_keep = batched_nms(
            minmax_box3d[nonempty_box_mask][:, [0, 1, 3, 4]],
            obj_scores[nonempty_box_mask], bbox_classes[nonempty_box_mask],
            self.test_cfg.nms_cfg)[1]

        if nms_keep.shape[0] > self.test_cfg.max_output_num:
            nms_keep = nms_keep[:self.test_cfg.max_output_num]

        # filter empty boxes and boxes with low score
        scores_mask = (obj_scores >= self.test_cfg.score_thr)
        nonempty_box_inds = torch.nonzero(
            nonempty_box_mask, as_tuple=False).flatten()
        nonempty_mask = torch.zeros_like(bbox_classes).scatter(
            0, nonempty_box_inds[nms_keep], 1)
        selected = (nonempty_mask.bool() & scores_mask.bool())

        if self.test_cfg.per_class_proposal:
            bbox_selected, score_selected, labels = [], [], []
            for k in range(sem_scores.shape[-1]):
                bbox_selected.append(bbox[selected].tensor)
                score_selected.append(obj_scores[selected])
                labels.append(
                    torch.zeros_like(bbox_classes[selected]).fill_(k))
            bbox_selected = torch.cat(bbox_selected, 0)
            score_selected = torch.cat(score_selected, 0)
            labels = torch.cat(labels, 0)
        else:
            bbox_selected = bbox[selected].tensor
            score_selected = obj_scores[selected]
            labels = bbox_classes[selected]

        return bbox_selected, score_selected, labels

    def _assign_targets_by_points_inside(self, bboxes_3d, points):
        """Compute assignment by checking whether point is inside bbox.

        Args:
            bboxes_3d (BaseInstance3DBoxes): Instance of bounding boxes.
            points (torch.Tensor): Points of a batch.

        Returns:
            tuple[torch.Tensor]: Flags indicating whether each point is
                inside bbox and the index of box where each point are in.
        """
        if isinstance(bboxes_3d, (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            points_mask = bboxes_3d.points_in_boxes_all(points)
            assignment = points_mask.argmax(dim=-1) # 如果不属于任何一个box也分配了？
        else:
            raise NotImplementedError('Unsupported bbox type!')

        return points_mask, assignment


    def _get_centreness_soft_target(self, gt_bboxes_3d, gt_labels_3d, points, enlarge):

        points_mask, assignment = self._assign_targets_by_points_inside(
            gt_bboxes_3d, points)
        (center_targets, size_targets, dir_class_targets,
         dir_res_targets) = self.bbox_coder.encode(gt_bboxes_3d, gt_labels_3d)
        center_targets = center_targets[assignment]
        size_res_targets = size_targets[assignment]
        mask_targets = gt_labels_3d[assignment]
        if enlarge:
            enlarged_gt_bboxes_3d = gt_bboxes_3d.enlarged_box(0.1)
            enlarged_points_mask, enlarged_assignment = self._assign_targets_by_points_inside(
                enlarged_gt_bboxes_3d, points)
            fg_mask = points_mask.max(1)[0] > 0
            enlarged_points_mask[fg_mask] = points_mask[fg_mask]
            enlarged_fg_mask = enlarged_points_mask.max(1)[0] > 0
            set_ignore = (fg_mask != enlarged_fg_mask)
            points_mask[set_ignore] = -1

        positive_mask = (points_mask.max(1)[0] > 0) # * dist_mask  # 有false就为false,该点同时满足在gtbox内 距离小于阈值
        negative_mask = (points_mask.max(1)[0] == 0)  # 不在gtbox内的点

        canonical_xyz = points - center_targets
        if self.bbox_coder.with_rot:
            # TODO: Align points rotation implementation of
            # LiDARInstance3DBoxes and DepthInstance3DBoxes
            canonical_xyz = rotation_3d_in_axis(
                canonical_xyz.unsqueeze(0).transpose(0, 1),
                -gt_bboxes_3d.yaw[assignment],
                axis=2).squeeze(1)
        # 与前后左右上下的距离
        distance_front = torch.clamp(
            size_res_targets[:, 0] - canonical_xyz[:, 0], min=0)
        distance_back = torch.clamp(
            size_res_targets[:, 0] + canonical_xyz[:, 0], min=0)
        distance_left = torch.clamp(
            size_res_targets[:, 1] - canonical_xyz[:, 1], min=0)
        distance_right = torch.clamp(
            size_res_targets[:, 1] + canonical_xyz[:, 1], min=0)
        distance_top = torch.clamp(
            size_res_targets[:, 2] - canonical_xyz[:, 2], min=0)
        distance_bottom = torch.clamp(
            size_res_targets[:, 2] + canonical_xyz[:, 2], min=0)

        centerness_l = torch.min(distance_front, distance_back) / torch.max(
            distance_front, distance_back)
        centerness_w = torch.min(distance_left, distance_right) / torch.max(
            distance_left, distance_right)
        centerness_h = torch.min(distance_bottom, distance_top) / torch.max(
            distance_bottom, distance_top)
        centerness_targets = torch.clamp(
            centerness_l * centerness_w * centerness_h, min=0)
        centerness_targets = centerness_targets.pow(1 / 3.0)
        centerness_targets = torch.clamp(centerness_targets, min=0, max=1)

        proposal_num = centerness_targets.shape[0]
        one_hot_centerness_targets = centerness_targets.new_zeros(
            (proposal_num, self.num_classes))
        one_hot_centerness_targets.scatter_(1, mask_targets.unsqueeze(-1), 1)
        centerness_targets = centerness_targets.unsqueeze(
            1) * one_hot_centerness_targets

        # 计算前景点占比 及各类点占比
        # cnt = (centerness_targets!=0).sum(dim=0)
        # my_string = '[{:.4f}, {:.4f}, {:.4f}],\n'.format(cnt[0], cnt[1], cnt[2])
        # with open('cnt_res.txt', 'a') as file:
        #     file.write(my_string)

        return centerness_targets, positive_mask, negative_mask
