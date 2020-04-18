#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import os
os.environ["TF_ALLOW_GPU_GROWTH"]='true'

from tensorflow.keras import layers

#%%

class rpn_subc(tf.keras.Model):
    def __init__(self):
        super(rpn_subc, self).__init__()
        
        ## define the blocks for add later just define the layers
        
        self.cov1_1 = layers.Conv2D(64, 3, activation = 'relu', padding ='same')
        self.cov1_2 = layers.Conv2D(64, 3, activation = 'relu', padding ='same')
        self.pool1 = layers.Maxpooling2D(2, strides = 2, padding = 'same')
        
        ## 
        self.cov2_1 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')
        self.cov2_2 = layers.Conv2D(128, 3, acitvation = 'relu', padding = 'same')
        self.pool2 =  layers.MaxPooling2D(2, strides = 2, padding = 'same')
        
        ##
        self.cov3_1 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same')
        self.cov3_2 = layers.Conv2D(256, 3, acitvation = 'relu', padding = 'same')
        self.cov3_3 = layers.Conv2D(256, 3, acitvation = 'relu', padding = 'same')
        self.pool3 =  layers.MaxPooling2D(2, strides = 2, padding = 'same')
        
        ##
        self.cov4_1 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same')
        self.cov4_2 = layers.Conv2D(512, 3, acitvation = 'relu', padding = 'same')
        self.cov4_3 = layers.Conv2D(512, 3, acitvation = 'relu', padding = 'same')
        self.pool4 =  layers.MaxPooling2D(2, strides = 2, padding = 'same')
        
        ##
        self.cov5_1 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same')
        self.cov5_2 = layers.Conv2D(512, 3, acitvation = 'relu', padding = 'same')
        self.cov5_3 = layers.Conv2D(512, 3, acitvation = 'relu', padding = 'same')
        self.pool5 =  layers.MaxPooling2D(2, strides = 2, padding = 'same')
        
        ##
        self.r_p_n_cov1 = layers.Conv2D(256, kernel_size = [5,2], 
                                        activation = tf.nn.relu,
                                        padding = 'same', use_bias = False)
        
        self.r_p_n_cov2 = layers.Conv2D(512, kernel_size = [5,2], 
                                        activation = tf.nn.relu,
                                        padding = 'same', use_bias = False)
        
        self.r_p_n_cov3 = layers.Conv2D(512, kernel_size = [5,2], 
                                        activation = tf.nn.relu,
                                        padding = 'same', use_bias = False)
        
        self.bboxes_cov = layers.Conv2D(36, kernel_size=[1,1], 
                                        padding = 'same', use_bias = False)
        
        self.scores_cov = layers.Conv2D(18, kernel_size = [1,1], 
                                        padding = 'same', use_bias=False)
        
    def call(self, x, training = False):
        ## Putting layer one after another
        h = self.cov1_1(x)
        h = self.cov1_2(h)
        h = self.pool1(h)
        
        h = self.cov2_1(h)
        h = self.cov2_2(h)
        h = self.pool2(h)
        
        h = self.cov3_1(h)
        h = self.cov3_2(h)
        h = self.cov3_3(h)
        h = self.pool3(h)
        ## one branched out
        pool3_p = tf.nn.max_pool2d(h, ksize = [1,2,2,1], strides = [1,2,2,1], 
                                   padding = 'same', name = 'pool3_prop')
        pool3_p =  self.r_p_n_cov1(pool3_p)
        
        h = self.cov4_1(h)
        h = self.cov4_2(h)
        h = self.cov4_3(h)
        h = self.pool4(h)
        pool4_p = self.r_p_n_cov2(h)
        
        h = self.cov5_1(h)
        h = self.cov5_2(h)
        h = self.cov5_3(h)
        
        pool5_p =  self.r_p_n_cov3(h)
        
        region_prop =  tf.concat([pool3_p, pool4_p, pool5_p], axis =-1)
        
        conv_cls_score = self.scores_cov(region_prop)
        
        cls_boxes =  self.bboxes_cov(region_prop)
        
        cls_scores = tf.reshape(conv_cls_score, [-1,45,60,9,2])
        cls_bboxes = tf.reshape(cls_boxes, [-1,45,60,9,4])
        
        return cls_scores, cls_bboxes
    
    
        