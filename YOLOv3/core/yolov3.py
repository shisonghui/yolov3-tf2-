#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg


# In[ ]:


NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)
STRIDES = np.array(cfg.YOLO.STRIDES)  #[8, 16, 32]
IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH


# In[22]:


def YOLOv3(input_data):
    #首先调用backbone的darknet53网络,输出data的shape为(None, , , 1024),
    #route_1shape( , , , 256), route_2shape( , , , 512)
    route_1, route_2, output_data = backbone.darknet53(input_data)
    #再来五次卷积操作
    conv = common.convolutional(output_data,(1, 1, 1024, 512))
    conv = common.convolutional(conv,(3, 3, 512, 1024))
    conv = common.convolutional(conv,(1, 1, 1024, 512))
    conv = common.convolutional(conv,(3, 3, 512, 1024))
    conv = common.convolutional(conv,(1, 1, 1024, 512))
    
    #尺度1
    conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3*(NUM_CALSS + 5)), activate=False, bn=False)
    
    
    conv = common.convolutional(conv, (1, 1,  512,  256))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_2], axis=-1)  #合并最后一维= 768
    conv = common.convolutional(conv, (1, 1, 768, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    #尺度2
    conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, bn=False)
    
    
    conv = common.convolutional(conv, (1,1,256,128))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)  #合并最后一维= 384
    conv = common.convolutional(conv, (1, 1, 384, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    #尺度3
    conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3*(NUM_CLASS +5)), activate=False, bn=False)
    
    return [conv_sbbox, conv_mbbox, conv_lbbox]
    


# In[ ]:


def decode(conv_output, i=0):  ##输入[conv_sbbox, conv_mbbox, conv_lbbox] each shape = 3*(NUM_CLASS +5)
    '''
    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
            contains (x, y, w, h, score, probability)
    '''
    conv_shape = tf.shape(conv_output)     ##????
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))
    
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2] #Bbox的 x, y #shape (batch_size,output_size,output_size,3,2)
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4] #Bbox的 w, h
    conv_raw_conf = conv_output[:, :, :, :, 4:5] #Bbox的 score
    conv_raw_prob = conv_output[:, :, :, :, 5: ] #Bbox的 probability
    #def tile(input, multiples, name=None) multiples：在指定的维度上复制原tensor的次数
    #tf.newaxis 插入新维度
    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size]) #shape (output_size,output_size)
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1]) #shape (output_size,output_size)
    
    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)  #shape (output_size,output_size,2)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1]) #shape (batch_size,output_size,output_size,3,2)
    xy_grid = tf.cast(xy_grid, tf.float32)
    
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    
    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


# In[45]:


#y = tf.tile(tf.range(5, dtype=tf.int32)[:, tf.newaxis], [1, 5]) #shape (output_size,output_size)
#x = tf.tile(tf.range(5, dtype=tf.int32)[tf.newaxis, :], [5, 1]) #shape (output_size,output_size)
    
#xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1) #shape (output_size,output_size,2)
#xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [3, 1, 1, 3, 1]) #shape (output_size,output_size,2)
#print(xy_grid)

