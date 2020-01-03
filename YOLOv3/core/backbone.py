#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import core.common as common

def darknet53(input_data):
    '''
    input_data是3通道的
    output_data是1024通道的
    '''
    x = common.convolutional(input_data, (3, 3, 3, 32))
    x = common.convolutional(x, (3, 3, 32, 64), downsample='True')
    
    for i in range(1):
        x = common.residual_block(x, 64, 32, 64)
        
    x = common.convolutional(x, (3, 3, 64, 128), downsample='True')
    
    for i in range(2):
        x = common.residual_block(x, 128, 64, 128)
        
    x = common.convolutional(x, (3, 3, 128, 256), downsample='True')
    
    for i in range(8):
        x = common.residual_block(x, 256, 128, 256)
        
    route_1 = x
    x = common.convolutional(x, (3, 3, 256, 512), downsample='True')
    
    for i in range(8):
        x = common.residual_block(x, 512, 256, 512)
        
    route_2 = x
    output_data = common.convolutional(x, (3, 3, 512, 1024), downsample='True')
    
    return route_1, route_2, output_data

