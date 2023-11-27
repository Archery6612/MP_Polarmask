import mmcv
import numpy as np
import torch
import math
from IPython import embed
import cv2
import pycocotools.mask as mask_util


def bbox2delta(proposals, gt, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0] + 1.0
    gh = gt[..., 3] - gt[..., 1] + 1.0

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox(rois,
               deltas,
               means=[0, 0, 0, 0],
               stds=[1, 1, 1, 1],
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(Tensor or ndarray): Shape (..., 4*k)
        img_shape(tuple): Image shape.

    Returns:
        Same type as `bboxes`: Flipped bboxes.
    """
    if isinstance(bboxes, torch.Tensor):
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.clone()
        flipped[:, 0::4] = img_shape[1] - bboxes[:, 2::4] - 1
        flipped[:, 2::4] = img_shape[1] - bboxes[:, 0::4] - 1
        return flipped
    elif isinstance(bboxes, np.ndarray):
        return mmcv.bbox_flip(bboxes, img_shape)


def bbox_mapping(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from the original image scale to testing scale"""
    new_bboxes = bboxes * scale_factor
    if flip:
        new_bboxes = bbox_flip(new_bboxes, img_shape)
    return new_bboxes


def bbox_mapping_back(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from testing scale to original image scale"""
    new_bboxes = bbox_flip(bboxes, img_shape) if flip else bboxes
    new_bboxes = new_bboxes / scale_factor
    return new_bboxes


def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def roi2bbox(rois):
    bbox_list = []
    img_ids = torch.unique(rois[:, 0].cpu(), sorted=True)
    for img_id in img_ids:
        inds = (rois[:, 0] == img_id.item())
        bbox = rois[inds, 1:]
        bbox_list.append(bbox)
    return bbox_list


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes - 1)]

'''bbox and mask 转成result mask要画图'''
def bbox_mask2result(bboxes, masks, labels,det_aprup, det_aplup, det_apldown ,det_aprdown,points,maskpred, indexpoints,num_classes,scale_points,img_meta,score = 0.3):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        masks (Tensor): shape (n, 2, 36)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    scale_points1 = torch.tensor([scale_points[0],scale_points[0]])
    ori_shape = img_meta['ori_shape']
    img_h, img_w, _ = ori_shape
    print(scale_points)
    mask_results = [[] for _ in range(num_classes - 1)]
    bboxes = bboxes.cpu().numpy()
    labels = labels.cpu().numpy()
    inds = np.where(bboxes[:, -1] > score)[0]
    bboxes = bboxes[inds]
    masks = masks[inds]
    print(masks)
    aprup_pred = det_aprup[inds]
    aplup_pred = det_aplup[inds]
    apldown_pred = det_apldown[inds]
    aprdown_pred = det_aprdown[inds]
    indexpoints = indexpoints[inds]
    res = torch.zeros(len(aprup_pred),2)
    remask = torch.zeros(len(aprup_pred),36)
    for i in range(len(aprup_pred)):
          points1 = points[int(indexpoints[i,3])]
          index = indexpoints[i,0]
          res[i] = points1[index.long()]
          remask[i] = maskpred[int(indexpoints[i,3])][index.long()]
    res = res / res.new_tensor(scale_points)
    scale_b = (bboxes[:,2] - bboxes[:,0]) / (bboxes[:,3] - bboxes[:,1])
    scale_rdown = (bboxes[:,2]- res[:,0].tolist()) / (-bboxes[:,0] + res[:,0].tolist())
    scale_rup = (bboxes[:,2]- res[:,0].tolist()) / (-bboxes[:,0] + res[:,0].tolist()) 
    scale_lup = (-bboxes[:,0] + res[:,0].tolist()) / (bboxes[:,2]- res[:,0].tolist())
    scale_ldown = (-bboxes[:,0] + res[:,0].tolist()) / (bboxes[:,2]- res[:,0].tolist())
    scale_rdown2 = (-bboxes[:,1] + res[:,1].tolist()) / (bboxes[:,3] - res[:,1].tolist())
    scale_rup2 = (bboxes[:,2] + bboxes[:,0])/2 - res[:,0].tolist()
    scale_lup2 = (bboxes[:,3] - res[:,1].tolist()) / (-bboxes[:,1] + res[:,1].tolist())
    scale_ldown2 = (-bboxes[:,1] + res[:,1].tolist()) / (bboxes[:,3] - res[:,1].tolist())

    scoreap = img_w / img_h 
    aprup, aplup, apldown, aprdown  =  ap2scale(points,aprup_pred,aplup_pred,apldown_pred,aprdown_pred,indexpoints,scale_b,scoreap,scale_rup,scale_lup,scale_ldown,scale_rdown,scale_rup2,scale_lup2,scale_ldown2,scale_rdown2,scale_points)
    det_aprup, det_aplup, det_apldown, det_aprdown,disaprup, disaplup, disapldown, disaprdown = ap2mask(points,maskpred,aprup, aplup, apldown, aprdown,indexpoints)
    det_aprup_c =  det_aprup/det_aprup.new_tensor(scale_points)
    det_aplup_c =  det_aplup/det_aplup.new_tensor(scale_points)
    det_apldown_c = det_apldown/det_apldown.new_tensor(scale_points)
    det_aprdown_c = det_aprdown/det_aprdown.new_tensor(scale_points)
    angles = torch.range(0, 350, 10).cuda() / 180 * math.pi
    img_shape = [768, 1280, 3]
    maskaprup = distance2mask(det_aprup_c, disaprup, angles,scale_points, max_shape=ori_shape)
    maskaplup = distance2mask(det_aplup_c, disaplup, angles,scale_points, max_shape=ori_shape)
    maskapldown = distance2mask(det_apldown_c, disapldown , angles, scale_points,max_shape=ori_shape)
    maskaprdown = distance2mask(det_aprdown_c, disaprdown, angles, scale_points,max_shape=ori_shape)
    det_aprup = maskaprup
    det_aplup = maskaplup
    det_apldown = maskapldown

    det_aprdown = maskaprdown
    det_aprup = torch.cat((det_aprup[:,:,36:36],det_aprup[:,:,0:9]),2)
    det_aplup = torch.cat((det_aplup[:,:,23:36],det_aplup[:,:,0:1]),2)
    det_aprdown = det_aprdown[:,:,6:22]
    det_apldown = det_apldown[:,:,15:31]
    for i in range(masks.shape[0]):
          anglelup = masktoangles(res[i],det_aplup[i])
          anglerup = masktoangles(res[i],det_aprup[i])
          angleldown = masktoangles(res[i],det_apldown[i])
          anglerdown = masktoangles(res[i],det_aprdown[i])
          anglelup = anglelup.cpu().numpy()
          anglerup = anglerup.cpu().numpy()
          angleldown = angleldown.cpu().numpy()
          anglerdown = anglerdown.cpu().numpy()
          indslup = np.where((360 >= anglelup) &  (anglelup >= 270))
          indsrup = np.where((90>= anglerup) &  (anglerup >= 0))
          indsldown = np.where((270 >= angleldown) &  (angleldown >= 180))
          indsrdown = np.where((180 >= anglerdown) &  (anglerdown >= 90))
          numsrdown = np.where((det_aprdown[i,:,indsrdown][0] > bboxes[i,2]) | (det_aprdown[i,:,indsrdown][1] < bboxes[i,1]))
          numsrup = np.where((det_aprup[i,:,indsrup][0] > bboxes[i,2]) | (det_aprup[i,:,indsrup][1] > bboxes[i,3]))
          numsldown = np.where((det_apldown[i,:,indsldown][0] < bboxes[i,0]) | (det_apldown[i,:,indsldown][1] < bboxes[i,1]))
          numslup = np.where((det_aplup[i,:,indslup][0] <= bboxes[i,0]) | (det_aplup[i,:,indslup][1] >= bboxes[i,3]))	
          det_rupmask = det_aprup[i,:,indsrup]
          det_lupmask = det_aplup[i,:,indslup]
          det_rdownmask = det_aprdown[i,:,indsrdown]
          det_ldownmask = det_apldown[i,:,indsldown]
          if(len(numsrdown[1]) > 3):
          	det_rdownmask = torch.Tensor(masks[i,:,9:18].unsqueeze(1).tolist())
          if(len(numsldown[1]) > 3):
          	det_ldownmask = torch.Tensor(masks[i,:,18:27].unsqueeze(1).tolist())
          if(len(numsrup[1]) > 3):
          	det_rupmask = torch.Tensor(masks[i,:,0:9].unsqueeze(1).tolist())
          if(len(numslup[1]) > 3):
          	det_lupmask = torch.Tensor(masks[i,:,27:36].unsqueeze(1).tolist())
          if(i == 0):
             apmask = single_finalmask(res[i],masks[i],det_rupmask,det_lupmask,det_ldownmask,det_rdownmask,masks.shape[2]).unsqueeze(0).tolist()
          else:
            apmask.append(single_finalmask(res[i],masks[i],det_rupmask,det_lupmask,det_ldownmask,det_rdownmask,masks.shape[2]).tolist())
          anglelup = masktoangles(res[i],det_aplup[i])
          anglerup = masktoangles(res[i],det_aprup[i])
          angleldown = masktoangles(res[i],det_apldown[i])
          anglerdown = masktoangles(res[i],det_aprdown[i])
          anglelup = anglelup.cpu().numpy()
          anglerup = anglerup.cpu().numpy()
          angleldown = angleldown.cpu().numpy()
          anglerdown = anglerdown.cpu().numpy()
          indslup = np.where((360 >= anglelup) &  (anglelup >= 270))
          indsrup = np.where((90 >= anglerup) &  (anglerup >= 0))
          indsldown = np.where((270 >= angleldown) &  (angleldown >= 180))
          indsrdown = np.where((180 >= anglerdown) &  (anglerdown >= 90))
          x_det_apall = torch.tensor(det_aprup[i][0][indsrup]).unsqueeze(0)
          y_det_apall = torch.tensor(det_aprup[i][1][indsrup]).unsqueeze(0)
          det_apall = torch.cat((x_det_apall,y_det_apall),0)
          x_det_apall = torch.tensor(det_aprdown[i][0][indsrdown]).unsqueeze(0)
          y_det_apall = torch.tensor(det_aprdown[i][1][indsrdown]).unsqueeze(0)
          ap = torch.cat((x_det_apall,y_det_apall),0)
          det_apall = torch.cat((det_apall,ap),1)
          x_det_apall = torch.tensor(det_apldown[i][0][indsldown]).unsqueeze(0)
          y_det_apall = torch.tensor(det_apldown[i][1][indsldown]).unsqueeze(0)
          ap = torch.cat((x_det_apall,y_det_apall),0)
          det_apall = torch.cat((det_apall,ap),1)
          x_det_apall = torch.tensor(det_aplup[i][0][indslup]).unsqueeze(0)
          y_det_apall = torch.tensor(det_aplup[i][1][indslup]).unsqueeze(0)
          ap = torch.cat((x_det_apall,y_det_apall),0)
          det_apall = torch.cat((det_apall,ap),1).unsqueeze(0)
          x_det_aprup = torch.tensor(det_aprup[i][0][indsrup]).unsqueeze(0)
          y_det_aprup = torch.tensor(det_aprup[i][1][indsrup]).unsqueeze(0)
          det_aprup1 = torch.cat((x_det_aprup,y_det_aprup),0).unsqueeze(0)
          x_det_apldown = torch.tensor(det_apldown[i][0][indsldown]).unsqueeze(0)
          y_det_apldown = torch.tensor(det_apldown[i][1][indsldown]).unsqueeze(0)
          det_apldown1 = torch.cat((x_det_apldown,y_det_apldown),0).unsqueeze(0)
          x_det_aprdown = torch.tensor(det_aprdown[i][0][indsrdown]).unsqueeze(0)
          y_det_aprdown = torch.tensor(det_aprdown[i][1][indsrdown]).unsqueeze(0)
          det_aprdown1 = torch.cat((x_det_aprdown,y_det_aprdown),0).unsqueeze(0)
          x_det_aplup = torch.tensor(det_aplup[i][0][indslup]).unsqueeze(0)
          y_det_aplup = torch.tensor(det_aplup[i][1][indslup]).unsqueeze(0)
          det_aplup1 = torch.cat((x_det_aplup,y_det_aplup),0).unsqueeze(0)
    for i in range(masks.shape[0]):
        im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask = [torch.Tensor(apmask[i]).transpose(1,0).unsqueeze(1).int().data.cpu().numpy()]
        im_mask = cv2.drawContours(im_mask, mask, -1,1,-1)
        rle = mask_util.encode(
            np.array(im_mask[:, :, np.newaxis], order='F'))[0]

        label = labels[i]
        mask_results[label].append(rle)
    if bboxes.shape[0] == 0:
        bbox_results = [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
    det_aplup1,det_aprup1,det_apldown1,det_aprdown1,apmask = [],[],[],[],[]
    return  bboxes, mask_results, bbox_results,res,det_aprup_c, det_aplup_c, det_apldown_c, det_aprdown_c,det_aplup1,det_aprup1,det_apldown1,det_aprdown1,apmask
def masktoangles(points, pos_mask_contour,ap = False):
        if(ap == True):
            ct = pos_mask_contour[:]
        else:
            ct = pos_mask_contour[:]
        x = ct[0] - points[0]
        y = ct[1] - points[1]
        angle = torch.atan2(x, y) * 180 / np.pi
        angle[angle < 0] += 360
        angle1 = angle.int()
        return angle1
def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    return torch.stack([x1, y1, x2, y2], -1)
def single_finalmask(points,centermask,aprupmask,aplupmask,apldownmask,aprdownmask,raynumbers):
            angle = masktoangles(points,centermask).cpu().numpy()
            rupangles = masktoangles(points,aprupmask)
            lupangles = masktoangles(points,aplupmask)
            ldownangles = masktoangles(points,apldownmask)
            rdownangles = masktoangles(points,aprdownmask)
            maxrup = torch.max(rupangles).cpu().numpy()
            minrup = max(torch.min(rupangles).cpu().numpy() ,0)
            maxlup = torch.max(lupangles).cpu().numpy()
            minlup = torch.min(lupangles).cpu().numpy() 
            maxrdown = torch.max(rdownangles).cpu().numpy() 
            minrdown = torch.min(rdownangles).cpu().numpy() 
            maxldown = min(torch.max(ldownangles).cpu().numpy() ,360)
            minldown = torch.min(ldownangles).cpu().numpy() 
            indexrup1 = np.where((minrup > angle) &  (angle >= 0))
            indexrup2 = np.where((minrdown > angle) &  (angle > maxrup))
            indexrdown = np.where((minldown > angle) &  (angle > maxrdown))
            indexldown = np.where((minlup > angle) &  (angle > maxldown))
            indexlup = np.where((360 >= angle) &  (angle > maxlup))
            centermask = torch.tensor(centermask.cpu().numpy())
            x_mask = centermask[0][indexrup1].unsqueeze(0)
            y_mask = centermask[1][indexrup1].unsqueeze(0)
            x_maskpoints = torch.cat((x_mask,aprupmask[0],centermask[0][indexrup2].unsqueeze(0),aprdownmask[0],centermask[0][indexrdown].unsqueeze(0),apldownmask[0],centermask[0][indexldown].unsqueeze(0),aplupmask[0],centermask[0][indexlup].unsqueeze(0)),1)
            y_maskpoints = torch.cat((y_mask,aprupmask[1],centermask[1][indexrup2].unsqueeze(0),aprdownmask[1],centermask[1][indexrdown].unsqueeze(0),apldownmask[1],centermask[1][indexldown].unsqueeze(0),aplupmask[1],centermask[1][indexlup].unsqueeze(0)),1)

            RP = torch.cat((x_maskpoints,y_maskpoints),0)
            return RP
def ap2mask(points,mask_pred,aprup, aplup, apldown, aprdown,indexpoints):
         det_aprup, det_aplup, det_apldown, det_aprdown,disaprup, disaplup, disapldown, disaprdown = [],[],[],[],[],[],[],[]
         det_aprup = torch.zeros(len(indexpoints),2)
         det_aplup = torch.zeros(len(indexpoints),2)
         det_apldown = torch.zeros(len(indexpoints),2)
         det_aprdown = torch.zeros(len(indexpoints),2)
         disaprup = torch.zeros(len(indexpoints),36)
         disaplup = torch.zeros(len(indexpoints),36)
         disapldown = torch.zeros(len(indexpoints),36)
         disaprdown = torch.zeros(len(indexpoints),36)
         for i in range(len(indexpoints)):
             det_aprup[i] = points[int(indexpoints[i,3])][aprup[i].long()]
             disaprup[i] = mask_pred[int(indexpoints[i,3])][aprup[i].long()]
             det_aplup[i] = points[int(indexpoints[i,3])][aplup[i].long()]
             disaplup[i] = mask_pred[int(indexpoints[i,3])][aplup[i].long()]
             det_apldown[i] = points[int(indexpoints[i,3])][apldown[i].long()]
             disapldown[i] = mask_pred[int(indexpoints[i,3])][apldown[i].long()]
             det_aprdown[i] = points[int(indexpoints[i,3])][aprdown[i].long()]
             disaprdown[i] = mask_pred[int(indexpoints[i,3])][int(aprdown[i])]
         return det_aprup, det_aplup, det_apldown, det_aprdown,disaprup, disaplup, disapldown, disaprdown
def ap2scale(points , apruppoints, apluppoints, apldownpoints, aprdownpoints, indexpoints,scale_b,scoreap,scale_rup,scale_lup,scale_ldown,scale_rdown,scale_rup2,scale_lup2,scale_ldown2,scale_rdown2,scale_points):
         aprup = torch.zeros(len(apruppoints))
         aplup = torch.zeros(len(apruppoints))
         apldown = torch.zeros(len(apruppoints))
         aprdown = torch.zeros(len(apruppoints))
         for i in range(len(apruppoints)):
            scale_w = indexpoints[i,1]
            ystride = indexpoints[i,2]
            points1 = points[int(indexpoints[i,3])]
            index = indexpoints[i,0]
            x_max = (int(index / ystride) + 1)*ystride - 1
            x_min = (int(index / ystride))*ystride
            if((index + math.floor(float(apruppoints[i][0]/scale_w)) + math.floor(float((apruppoints[i][1]/scale_w)))*ystride) < 0):
                aprup[i] = index
            elif((index  + math.floor(float(apruppoints[i][0]/scale_w)) + math.floor(float((apruppoints[i][1]/scale_w)))*ystride) > (len(points1)-1)):
                aprup[i] = index
            elif(x_min > (index  + round(float((apruppoints[i][0]*0.6 - scale_rup2[i])/scale_w)))):
                aprup[i]  = index
            elif((index  + round(float((apruppoints[i][0]*0.6 - scale_rup2[i])/scale_w))) > x_max):
                 aprup[i]  = x_max + math.floor(float((apruppoints[i][1]/scale_w)))*(ystride)
            else:
                weight = (apruppoints[i][0])*scale_points[1]/(apruppoints[i][1])/scale_points[0]
                if(scale_rup[i]>=1):
                 	aprup[i]  = index  + max(round(float((apruppoints[i][0]*0.6 - scale_rup2[i])/scale_w))-2,0) + math.floor(float((apruppoints[i][1]/scale_w)))*(ystride) 
                else:
                       aprup[i]  = index  + max(math.floor(float((apruppoints[i][0]*0.6 - scale_rup2[i])/scale_w))-2,0) + math.floor(float((apruppoints[i][1]/scale_w)))*(ystride)
            if((index  - math.floor(float(apluppoints[i][0]/scale_w)) + math.floor(float((apluppoints[i][1]/scale_w)))*ystride) < 0):
                 aplup[i] = index
            elif((index  - math.floor(float(apluppoints[i][0]/scale_w)) + math.floor(float((apluppoints[i][1]/scale_w)))*ystride) > (len(points1)-1)):
                 aplup[i] = index
            elif(x_min > (index  - round(float((apluppoints[i][0]*0.6 + scale_rup2[i])/scale_w)))):
                aplup[i]  = x_min + math.floor(float((apluppoints[i][1]/scale_w)))*(ystride)
            elif((index  - round(float((apluppoints[i][0]*0.6 + scale_rup2[i])/scale_w))) > x_max):
                aplup[i]  = index
            else:
                weight = (apluppoints[i][0])*scale_points[1]/(apluppoints[i][1])/scale_points[0]
                if(scale_lup[i]>=1):
                 	aplup[i]  = index - max(round(float((apluppoints[i][0]*0.6+scale_rup2[i])/scale_w)),0) + math.floor(float((apluppoints[i][1]/scale_w)))*(int(ystride))
                else:
                 	aplup[i]  = index - max(round(float((apluppoints[i][0]*0.6+scale_rup2[i])/scale_w)),0) + math.floor(float((apluppoints[i][1]/scale_w))-1)*(int(ystride))                   
            if((index - math.floor(float(apldownpoints[i][0]/scale_w)) - math.floor(float((apldownpoints[i][1]/scale_w)))*ystride) < 0):
                apldown[i] = index
            elif((index  - math.floor(float(apldownpoints[i][0]/scale_w)) - math.floor(float((apldownpoints[i][1]/scale_w)))*ystride) > (len(points1)-1)):
                apldown[i] = index
            elif(x_min > (index  - round(float((apldownpoints[i][0]*0.6 + scale_rup2[i])/scale_w)))):
                apldown[i]  = x_min - math.floor(float((apldownpoints[i][1]/scale_w)))*(ystride)
            elif((index  - round(float((apldownpoints[i][0]*0.6 + scale_rup2[i])/scale_w))) > x_max):
                apldown[i]  = index
            else:
                weight = (apldownpoints[i][0])*scale_points[1]/(apldownpoints[i][1])/scale_points[0]
                if(scale_ldown[i]>=1):
                 	apldown[i]  = index  - max(round(float((apldownpoints[i][0]*0.6 + scale_rup2[i])//scale_w)),0) - math.floor(float((apldownpoints[i][1]/scale_w)))*(ystride) 
                else:
                 	apldown[i]  = index  - max(math.floor(float((apldownpoints[i][0]*0.6 + scale_rup2[i])/scale_w)),0) - math.floor(float((apldownpoints[i][1]/scale_w)))*(ystride)                  
            if((index  - math.floor(float(aprdownpoints[i][0]/scale_w)) + math.floor(float((aprdownpoints[i][1]/scale_w)))*ystride) < 0):
                aprdown[i] = index
            elif((index  + math.floor(float(aprdownpoints[i][0]/scale_w)) - math.floor(float((aprdownpoints[i][1]/scale_w)))*ystride) > (len(points1)-1)):
                aprdown[i] = index
            elif(x_min > (index  + round(float((aprdownpoints[i][0]*0.6 - scale_rup2[i])/scale_w)))):
                 aprdown[i]  = index
            elif((index  + round(float((aprdownpoints[i][0]*0.6 - scale_rup2[i])/scale_w))) > x_max):
                aprdown[i]  = x_max - math.floor(float((aprdownpoints[i][1]/scale_w)))*(ystride)
            else:
                if(scale_rdown[i]>=1):
                 	aprdown[i]  = index  + max(round(float((aprdownpoints[i][0]*0.6 - scale_rup2[i])/scale_w)),0) - math.floor(float((aprdownpoints[i][1]/scale_w))+1)*(ystride)
                else:
                    aprdown[i]  = index  + max(math.floor(float((aprdownpoints[i][0]*0.6 - scale_rup2[i])/scale_w)),0) - math.floor(float((aprdownpoints[i][1]/scale_w)))*(ystride) 
                    weight = (aprdownpoints[i][0])*scale_points[1]/(aprdownpoints[i][1])/scale_points[0]
         return  aprup, aplup, apldown, aprdown   
def distance2mask(points, distances, angles, scale_points,max_shape=None):
    '''Decode distance prediction to 36 mask points
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 36,from angle 0 to 350.
        angles (Tensor):
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded masks.
    '''
    num_points = points.shape[0]
    res = torch.zeros((len(points),2,36))
    for i in range(len(points)):
        x_points = points[i,0].repeat(1,36)
        y_points = points[i,1].repeat(1,36)
        c_x, c_y = x_points, y_points
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        sin = torch.tensor(sin.cpu().numpy())
        cos = torch.tensor(cos.cpu().numpy())
        x = distances[i] * sin / scale_points[0] + c_x 
        y = distances[i] * cos /scale_points[1] + c_y
        if max_shape is not None:
           x = x.clamp(min=0, max=max_shape[1] - 1)
           y = y.clamp(min=0, max=max_shape[0] - 1)
        res[i][0] = x
        res[i][1] = y
    return res





