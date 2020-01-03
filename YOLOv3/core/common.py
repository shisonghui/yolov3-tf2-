#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf


# In[ ]:


class BatchNormalization(tf.keras.layers.BatchNormalization):
    '''
    “frozen state”和“inference mode”是两个独立的概念。
    layer.trainable = False用于冻结该层，
    因此该层将在“推理模式”中使用存储的移动var和均值，
    并且gama和beta将不会更新！
    '''
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)#逻辑和
        return super().call(x, training)


# In[ ]:


def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True):
    '''
    filters_shape=(kernel_size, kernel_size,  input_channel, filter_num)
    a = tf.constant([[[[1,2,3],[4,5,6],[7,8,9]],
                 [[1,2,3],[4,5,6],[7,8,9]],
                 [[1,2,3],[4,5,6],[7,8,9]]]])
    input_layer = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))(a)
    '''
    #是否下采样
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))(input_layer)#输入的shape(None,h,w,c)
        padding = 'valid'                                                      #输出shape(None,h+1,w+1,c)
        strides = 2
    else:
        strides = 1
        padding = 'same'
        
    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0],
                                  strides=strides,padding=padding,use_bias=not bn,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                 bias_initializer=tf.constant_initializer(0.))(input_layer)
    if bn: conv = BatchNormalization()(conv)
    if activate: conv = tf.nn.leaky_relu(conv, alpha=0.1)
        
    return conv


# In[25]:


def residual_block(input_layer, input_channel, filter_num1, filter_num2):
    '''
    残差块
    '''
    short_cut = input_layer
    conv = convolutional(input_layer, (1, 1, input_channel, filter_num1))
    conv = convolutional(conv       , (3, 3, filter_num1,   filter_num2))

    residual_output = short_cut + conv
    return residual_output


# In[ ]:


def upsample(input_layer):
    '''
    上采样
    '''
    return tf.image.resize(input_layer, (input_layer[1]*2, input_layer[2]*2), method='nearest')

