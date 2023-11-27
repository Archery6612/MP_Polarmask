from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import torch.nn as nn

from mmdet.core import bbox2result, bbox_mask2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from IPython import embed
import time
import torch


@DETECTORS.register_module
class PolarMask(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PolarMask, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)
     ##  add _ap_polarmask_targets and _ap_center_targets
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      _gt_labels=None,
                      _gt_bboxes=None,
                      _gt_masks=None,
                      _ap_centerness_targets =None,
                      _ap_masspoint_rup_targets=None,
                      _ap_masspoint_lup_targets=None,
                      _ap_masspoint_ldown_targets=None,
                      _ap_masspoint_rdown_targets=None):

        if _gt_labels is not None:
            extra_data = dict(_gt_labels=_gt_labels,
                              _gt_bboxes=_gt_bboxes,
                              _gt_masks=_gt_masks,
                              _ap_centerness_targets = _ap_centerness_targets,
                              _ap_masspoint_rup_targets=_ap_masspoint_rup_targets,
                              _ap_masspoint_lup_targets=_ap_masspoint_lup_targets,
                              _ap_masspoint_ldown_targets=_ap_masspoint_ldown_targets,
                              _ap_masspoint_rdown_targets=_ap_masspoint_rdown_targets)
        else:
            extra_data = None


        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)

        losses = self.bbox_head.loss(
            *loss_inputs,
            gt_masks = gt_masks,
            gt_bboxes_ignore=gt_bboxes_ignore,
            extra_data=extra_data
        )
        return losses


    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        print(len(bbox_list[0]))
        results = [
            bbox_mask2result(det_bboxes, det_masks, det_labels,det_aprup, det_aplup, det_apldown ,det_aprdown,points,det_maskpred,det_indexpoints, self.bbox_head.num_classes,scale_points, img_meta[0],score = 0.35)
            for det_bboxes, det_labels, det_masks,det_aprup, det_aplup, det_apldown ,det_aprdown,points,det_maskpred,det_indexpoints,scale_points in bbox_list]
        bboxes = results[0][0]
        bbox_results = results[0][2]
        mask_results = results[0][1]
        res = results[0][3]
        det_aprup_c = results[0][4]
        det_aplup_c = results[0][5]
        det_apldown_c = results[0][6]
        det_aprdown_c = results[0][7]
        det_aplup1 = results[0][8]
        det_aprup1 = results[0][9]
        det_apldown1 = results[0][10]
        det_aprdown1 = results[0][11]
        apmask = results[0][12]
        return bboxes, mask_results, bbox_results,res,det_aprup_c, det_aplup_c, det_apldown_c, det_aprdown_c,det_aplup1,det_aprup1,det_apldown1,det_aprdown1,apmask
