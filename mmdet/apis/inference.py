import warnings
import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.runner import load_checkpoint
import os
from mmdet.core import get_classes
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
from mmdet.models import build_detector


def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    device = next(model.parameters()).device  # model device
    if not isinstance(imgs, list):
        return _inference_single(model, imgs, img_transform, device)
    else:
        return _inference_generator(model, imgs, img_transform, device)


def _prepare_data(img, img_transform, cfg, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])


def _inference_single(model, img, img_transform, device):
    #plt.figure()
    img = mmcv.imread(img) 
    #fig = plt.gcf()
    #print(img)
    cv2.imwrite(os.path.join('/home/hscc/PolarMask/demo','img.jpg'),img)
    #img.savefig("img.jpg")
    #plt.imshow(img)
    data = _prepare_data(img, img_transform, model.cfg, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def _inference_generator(model, imgs, img_transform, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, device)


# TODO: merge this method with the one in BaseDetector
def show_result(img,
                result,
                class_names,
                score_thr=0.35,
                wait_time=0.1,
                show=True,
                out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
         bboxes, segm_result,bbox_results,res,det_aprup_c, det_aplup_c, det_apldown_c, det_aprdown_c,det_aplup1,det_aprup1,det_apldown1,det_aprdown1,apmask = result #,res,det_aprup_c, det_aplup_c, det_apldown_c, det_aprdown_c,det_aplup1,det_aprup1,det_apldown1,det_aprdown1,apmask 
    else:
        bboxes, segm_result,bbox_results= result, None
    # draw segmentation masks
    bboxes = np.vstack(bbox_results)
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where((np.vstack(bbox_results)[:, -1]) > score_thr)[0]
        for i in range(len(inds)):
            #print(i)
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            #print(len(mask[0]))
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_results)
    ]
    labels = np.concatenate(labels)
    det_aplup1 = det_aplup1.int()
    det_aprup1 = det_aprup1.int()
    det_apldown1 = det_apldown1.int()
    det_aprdown1 = det_aprdown1.int()
    #for i in range(det_aprup_c.shape[0]):
        #cv2.circle(img, (int(res[i][0]),int(res[i][1])), 1 , (255,255,255) , 8)
        #for j in range(len(apmask[i][0])):
        	#cv2.circle(img,(int(apmask[i][0][j]),int(apmask[i][1][j])), 1, (255,255,255), 6)
        #cv2.circle(img, (int(det_aprup_c[i][0]),int(det_aprup_c[i][1])), 1 , (0,0,255) , 8)
        #for j in range(len(det_aprup1[i][0])):
        	#cv2.circle(img,(det_aprup1[i][0][j].tolist(),det_aprup1[i][1][j].tolist()), 1, (0,0,255), 6)
        #cv2.circle(img, (int(det_aplup_c[i][0]),int(det_aplup_c[i][1])), 1 , (0,255,0) , 8)
        #for j in range(len(det_aplup1[i][0])):
        	#cv2.circle(img,(det_aplup1[i][0][j].tolist(),det_aplup1[i][1][j].tolist()), 1, (0,255,0), 6)
        #cv2.circle(img, (int(det_aprdown_c[i][0]),int(det_aprdown_c[i][1])), 1 , (255,0,0) , 8)
        #for j in range(len(det_aprdown1[i][0])):
        	#cv2.circle(img,(det_aprdown1[i][0][j].tolist(),det_aprdown1[i][1][j].tolist()), 1, (255,0,0), 6)
        #cv2.circle(img, (int(det_apldown_c[i][0]),int(det_apldown_c[i][1])), 1 , (0,0,0) , 8)
    mmcv.imshow_det_bboxes(
        img,
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    if not (show or out_file):
        return img


def show_result_pyplot(img,
                       result,
                       class_names,
                       score_thr=0.35,
                       fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    img1 = show_result(
        img, result, class_names, score_thr=score_thr, show=False)
    #print(img1)
    #img = mmcv.bgr2rgb(img1)
    cv2.imwrite(os.path.join('/home/hscc/PolarMask/demo','result.jpg'),img1)
    #plt.figure(figsize=fig_size)
    #plt.savefig("result.png")
    #plt.imshow(mmcv.bgr2rgb(img))