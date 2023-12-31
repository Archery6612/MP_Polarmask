import torch

from mmdet.ops.nms import nms_wrapper
from IPython import embed


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i - 1,
                                           dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels

def multiclass_nms_with_mask(multi_bboxes,
                   multi_scores,
                   multi_masks,
                   mlvl_aprupmasks, 
                   mlvl_aplupmasks, 
                   mlvl_apldownmasks, 
                   mlvl_aprdownmasks,
                   mlvl_points,
                   mlvl_maskpred,
                   indexpoint,
                   score_thr,
                   nms_cfg,
                   scalefactor,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels, masks,aprupmasks,aplupmasks,apldownmasks,aprdownmasks,points,maskspred,indexpoints = [], [], [], [], [], [], [],[],[],[]
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
            _masks  = multi_masks[cls_inds, :]
            _aprupmasks  = mlvl_aprupmasks[cls_inds, :]
            _aplupmasks  =  mlvl_aplupmasks[cls_inds, :]
            _apldownmasks  = mlvl_apldownmasks[cls_inds, :]
            _aprdownmasks  = mlvl_aprdownmasks[cls_inds, :]
            _indexpoint = indexpoint[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, index = nms_op(cls_dets, **nms_cfg_)
        cls_masks = _masks[index]
        cls_aprupmasks = _aprupmasks[index]
        cls_aplupmasks = _aplupmasks[index]
        cls_apldownmasks = _apldownmasks[index]
        cls_aprdownmasks = _aprdownmasks[index]
        cls_indexpoint = _indexpoint[index]
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i - 1,
                                           dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
        masks.append(cls_masks)
        aprupmasks.append(cls_aprupmasks)
        aplupmasks.append(cls_aplupmasks)
        apldownmasks.append(cls_apldownmasks)
        aprdownmasks.append(cls_aprdownmasks)
        indexpoints.append(cls_indexpoint)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        masks = torch.cat(masks)
        aprupmasks = torch.cat(aprupmasks)
        aplupmasks = torch.cat(aplupmasks)
        apldownmasks = torch.cat(apldownmasks)
        aprdownmasks = torch.cat(aprdownmasks)
        indexpoints = torch.cat(indexpoints)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
            masks = masks[inds]
            aprupmasks = aprupmasks[inds]
            aplupmasks = aplupmasks[inds]
            apldownmasks = apldownmasks[inds]
            aprdownmasks = aprdownmasks[inds]
            indexpoints = indexpoints[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        masks = multi_bboxes.new_zeros((0, 2, 36))
        aprupmasks = multi_bboxes.new_zeros((0, 2, 36))
        aplupmasks = multi_bboxes.new_zeros((0, 2, 36))
        apldownmasks = multi_bboxes.new_zeros((0, 2, 36))
        aprdownmasks = multi_bboxes.new_zeros((0, 2, 36))
        indexpoints = multi_bboxes.new_zeros((0, 4))
    return bboxes, labels, masks,aprupmasks,aplupmasks,apldownmasks,aprdownmasks,mlvl_points,mlvl_maskpred,indexpoints
