#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import random
import colorsys
import numpy as np
from core.config import cfg


# In[ ]:


def load_weights(model, weights_file):
    """
    I agree that this code is very ugly, but I don’t know any better way of doing it.
    """
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(75):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in [58, 66, 74]:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in [58, 66, 74]:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


# In[ ]:


def read_calss_names(classes_path):
    '''loads class name from a file'''
    names = {}
    with open(classes_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.stripp('\n')
    return names


# In[2]:


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','),dtype=np.float32)
    return anchors.reshape(3,3,2)


# In[ ]:


def image_preporcess(image, target_size, gt_boxes=None):
    '''
    gt_boxes.shape=(None, 5)  (x, y, w, h, class)
    array([[358, 222, 400, 264,   0],
       [208, 147, 264, 203,   1],
       [347, 313, 375, 341,   3],
       [115,  49, 171, 105,   1]])
    '''
    ih, iw    = target_size  #要改的大小
    h,  w, _  = image.shape  #图像实际大小
    
    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))  #改变图像大小
    
    image_paded = np.full(shape=[ih, iw, 3], fill_value= 128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.  #shape (ih,iw,3)
    
    if gt_boxes is None:
        return image_paded
    else:   #bboxes的大小也要相应的改变
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


# In[ ]:


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)
    
    pred_xywh = pred_bbox[:,0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]
    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio    #1,3列
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio    #2,4列
    
    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))#找到超出范围的边框
    pred_coor[invalid_mask] = 0  #删除超出范围的边框
    
    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
    
    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]
    
    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


# In[ ]:


def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)  #1.1920929e-07

    return ious


# In[ ]:


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    '''
    非极大值抑制
    bboxes: (xmin, ymin, xmax, ymax, score, class) 边框左上角和右下角的点
    '''
    classes_in_img = list(set(bboxes[:,5]))  #图像中有多少种类别出现，用set集合
    best_bboxes = []  #用来装最好的边框
    
    for cls in classes_in_img:    #对每一类单独进行分析
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[:max_ind],cls_bboxes[max_ind+1 :]])  #cls_bboxes除去分数最大的那个边框
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)
        
            assert method in ['nms', 'soft-nms']
            
            if method == 'nms':
                iou_mask = iou > iou_threshold    #小于iou_threshold的边框下一次循环继续处理
                weight[iou_mask] = 0.0
                
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
                
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight   #大于iou_threshold的那些边框的分数将被变为0
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]    #除去iou大于iou_threshold的那些边框
            
    return best_bboxes


# In[ ]:


def draw_bbox(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image


# In[33]:


'''
import numpy as np

ih, iw    = 315,315  #要改的大小
h,  w  = 416,416  #图像实际大小

scale = min(iw/w, ih/h)
print(scale)
nw, nh  = int(scale * w), int(scale * h)
print(nw)
image_resized = np.ones((nw, nh))  #改变图像大小

image_paded = np.full(shape=[ih, iw], fill_value= 128.0)
dw, dh = (iw - nw) // 2, (ih-nh) // 2
image_paded[dh:nh+dh, dw:nw+dw] = image_resized
'''


# In[88]:


#anno = "E:\TensorFlow2.0-Examples-master\TensorFlow2.0-Examples-master\4-Object_Detection\YOLOV3\data\dataset\train\000001.jpg 358,222,400,264,0 208,147,264,203,1 347,313,375,341,3 115,49,171,105,1"
#line = anno.strip().split(' ')
#gt_boxes = np.array([list(map(int, box.split(','))) for box in line[1:]])


# In[89]:


#gt_boxes


# In[90]:


#gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
#gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh


# In[91]:


#gt_boxes


# In[53]:


#strides = np.array([8, 16, 32])
#for bbox in gt_boxes:
#    bbox_coor = bbox[:4]
#    bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
#    print(bbox_xywh)
#    bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]
 #   print(bbox_xywh_scaled)


# In[50]:


#bbox_xywh.shape


# In[78]:


#gt_boxes


# In[72]:


#invalid_mask = np.logical_or((gt_boxes[:, 0] < gt_boxes[:, 2]), (gt_boxes[:, 1] > gt_boxes[:, 3]))


# In[86]:


#invalid_mask = [ False,  True,  False,  True]


# In[64]:


#(gt_boxes[:, 0] < gt_boxes[:, 2])


# In[65]:


#(gt_boxes[:, 1] > gt_boxes[:, 3])


# In[81]:


#gt_boxes[invalid_mask] = 0


# In[87]:


#gt_boxes


# In[92]:


#gt_boxes = gt_boxes[invalid_mask]


# In[93]:


#gt_boxes


# In[ ]:




