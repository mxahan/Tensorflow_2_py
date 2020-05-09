#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 02:14:10 2020

@author: zahid
"""
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import Model,layers


#%%
class ConvBnRelu(Model):
    def __init__(self, ch, kernel_size = 3, strides= 1, padding = 'same'):
        super(ConvBnRelu, self).__init__()
        
        self.model = keras.models.Sequential([
            layers.Conv2D(ch, kernel_size = kernel_size, strides= strides, padding=padding),
            layers.BatchNormalization(),
            layers.ReLU()
            ])
    def call(self, x, training=None):
        x = self.model(x, training=training)
        return x
    

class ConvTrBnRelu(Model):
    def __init__(self, ch, kernel_size = 3, strides= (1,1), padding = 'same'):
        super(ConvTrBnRelu, self).__init__()
        
        self.model = keras.models.Sequential([
            layers.Conv2DTranspose(filters=ch, kernel_size=kernel_size, strides= strides, padding=padding),
            layers.BatchNormalization(),
            layers.LeakyReLU()
            ])
    def call(self, x, training=None):
        x = self.model(x, training=training)
        return x

class Discriminator(Model):
    def __init__(self, classnumber):
        super(Discriminator, self).__init__()
        self.conv1 = ConvBnRelu(ch = 32, kernel_size=3, strides= 2) # size, 128*128*32
        self.conv2 = ConvBnRelu(ch = 32, kernel_size = 3 , strides = 2) # 64,64,32
        self.conv3 = ConvBnRelu(ch = 16, kernel_size = 3 , strides = 2) # 32,32,16
        self.conv4 = ConvBnRelu(ch = 8, kernel_size = 3 , strides = 2) # 16,16,8
        self.flatten = layers.Flatten()
        # self.fc1 = layers.Dense(1024)
        # self.bn1 = layers.BatchNormalization()
        self.fc2 = layers.Dense(512)
        self.bn2 = layers.BatchNormalization()
        self.out = layers.Dense(classnumber+1)
        
    
        
    def call(self, x, training = None):
        x = tf.reshape(x, [-1, 256, 256, 3])
        x = self.conv1(x, training = training)
        x = self.conv2(x, training = training)
        x = self.conv3(x, training = training)
        x = self.conv4(x, training = training)
        
        x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.bn1(x, training = training)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = self.bn2(x, training = training)
        x= tf.nn.relu(x)
        x =  self.out(x)
        return x
        
        


class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = layers.Dense(4*4*512)
        self.conv1 = ConvTrBnRelu(ch = 512, kernel_size = 3, strides =(2,2)) # size 8,8, 128
        self.conv2 = ConvTrBnRelu(ch = 256, kernel_size = 3, strides =(2,2)) # size 16,16, 64
        self.conv3 = ConvTrBnRelu(ch = 128, kernel_size = 3, strides =(2,2)) # size 32,32, 32
        self.conv4 = ConvTrBnRelu(ch = 32, kernel_size = 3, strides =(2,2)) # size 64,64, 16
        self.conv5 = ConvTrBnRelu(ch = 32, kernel_size = 3, strides =(2,2)) # size 128,128, 8
        self.conv6 = layers.Conv2DTranspose(filters = 3, kernel_size = 3,
                                            padding= 'same', strides =(2,2), 
                                            activation = 'tanh') # size 256, 256, 3
        
    def call(self, x, training = None):
        x = tf.reshape(x, shape= [-1, 500])
        x = self.fc1(x)
        x =  tf.nn.leaky_relu(x)
        x = tf.reshape(x ,[-1, 4,4,512])
        x = self.conv1(x, training =  training)
        x = self.conv2(x, training =  training)
        x = self.conv3(x, training =  training)
        # print(x.shape)
        x = self.conv4(x, training =  training)
        x = self.conv5(x, training =  training)
        x = self.conv6(x)
        
        return x
