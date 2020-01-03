#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
cv2.imread()接口读图像，读进来直接是BGR 格式数据格式在 0~255
需要特别注意的是图片读出来的格式是BGR，不是我们最常见的RGB格式，颜色肯定有区别
cv2.cvtColor(p1,p2) 是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式
cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片
'''


# In[16]:


import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg


# In[13]:


class Dataset(object):
    def __init__(self, dataset_type):
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH #"./data/dataset/yymnist_train.txt"
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE  #416  544
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE  #4  2
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG  #True  False
        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE  #416
        self.strides = np.array(cfg.YOLO.STRIDES)      #[8, 16, 32]
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)    #"./data/classes/coco.names"注意格式是.names  返回的是字典names
        self.num_classes = len(self.classes)  #类别数目
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS)) #return anchors.reshape(3,3,2)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE  # 3
        self.max_bbox_per_scale = 150
        self.annotations = self.load_annotations(dataset_type)  #注解
        self.num_samples = len(self.annotations)    #注解的长度就是样本的数量
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))  #batch的数目 #np.ceil()计算大于等于该值的最小整数
        self.batch_count = 0
    def load_annotations(self, dataset_type):
        '''下载注解'''
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()    #读取整个文件（f）所有行，保存在一个列表(list)变量中，每行作为一个元素，但读取大文件会比较占内存。
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)   #打乱顺序
        return annotations
    
    def __iter__(self):
        '''定义__iter__方法，该方法返回迭代器'''
        return self
    
    def __next__(self):
        '''定义迭代器所需要的__next__方法，返回迭代器的下一个元素'''
        with tf.device('/cpu:0'): #使用 tf.device() 指定模型运行的具体设备，可以指定运行在GPU还是CUP上，以及哪块GPU上。
                                  #tensorflow中不同的GPU使用/gpu:0和/gpu:1区分，而CPU不区分设备号，统一使用 /cpu:0
            self.train_input_size = np.random.choice(self.train_input_sizes)  #随机取0-415之间的数
            self.train_output_sizes = self.train_input_size // self.strides   #返回一维数组，长度是3

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3), dtype=np.float32)

            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)  #分析注解,image.shape=(self.train_input_size, self.train_input_size,3)且归一化 /255
                                                                       #bboxes 也相应的改变了
                    #处理边界框，得到三种尺寸下： small, medium, large的样本标签y和其对应的边界框
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes  #元组（标签，边界框数据）
                batch_medium_target  = batch_label_mbbox, batch_mbboxes
                batch_larger_target  = batch_label_lbbox, batch_lbboxes

                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration
    
    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, bboxes

     def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes
    
    def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes
    
    def parse_annotation(self, annotations):
        
        line = annotations.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist..." %image_path)
        image = cv2.imread(image_path)
        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])
        
        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image),np.copy(bboxes))  #np.copy 浅拷贝 随机水平镜像
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))  #随机裁剪
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))  #随机翻译
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #转换图像格式为RGB
        image, bboxes = utils.image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
        
        return image, bboxes
    
    def bbox_iou(self, boxes1, boxes2):
        '''
        boxes1 /2 shape (3, 4) ，3代表每个尺度有三个anchor，每个anchor是四维的（x,y,w,h）
        '''
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]  #面积=宽*高
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]
        #求两个框的交集面积  inter_area
        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        
        #并集面积  union_area
        union_area = boxes1_area + boxes2_area - inter_area
        
        return inter_area / union_area  #两个框boxes1, boxes2的交并比
    
    def preprocess_true_boxes(self, bboxes):
        
        
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                         5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))
        
        for bbox in bboxes:
            bbox_coor = bbox[:,4]
            bbox_class_ind = bbox[4]
            
            #对类别进行onehot编码
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
            
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            
            iou = []
            exist_positive = False
            
            for i in range(3):  #共三种尺寸： small, medium, large
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]   #事前聚类得到的anchors宽高，共9个，每个尺寸3个
                
                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)  #计算边界框和anchor的交并比
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3
                
                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    #样本标签 (x, y, w, h, score, probability)
                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh  #存储所有边界框的数据，且容量不得大于self.max_bbox_per_scale
                    bbox_count[i] += 1

                    exist_positive = True
                
            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
    
    def __len__(self):
        return self.num_batchs


# In[9]:


#import numpy as np
#x = np.random.choice(416)


# In[12]:


#x // np.array([1,2,3])


# In[13]:



#a = [1, 2, ['a']]
#b = np.copy(a)


# In[18]:


#a[2].append('b')
#a.append(3)


# In[19]:


#a


# In[20]:


#b


# In[24]:


#bbox_count = np.zeros(80)


# In[25]:


#bbox_count.shape


# In[26]:


#onehot = np.zeros(10, dtype=np.float)
#onehot[5] = 1.0
#uniform_distribution = np.full(10, 1.0 / 10)
#deta = 0.01
#smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution


# In[27]:


#smooth_onehot


# In[28]:


#onehot


# In[ ]:




