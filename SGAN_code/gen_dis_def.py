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
            layers.Conv2D(ch, kernel_size = kernel_size, strides= strides, padding=padding,
                          kernel_initializer='random_normal'),
            layers.ReLU(), 
            layers.BatchNormalization()
            ])
    def call(self, x, training =  None):
        x = self.model(x, training = training)
        return x
    

class ConvTrBnRelu(Model):
    def __init__(self, ch, kernel_size = 3, strides= (1,1), padding = 'same'):
        super(ConvTrBnRelu, self).__init__()
        
        self.model = keras.models.Sequential([
            layers.Conv2DTranspose(filters=ch, kernel_size=kernel_size, strides= strides,
                                   padding=padding, kernel_initializer='random_normal'),
            layers.LeakyReLU(),
            layers.BatchNormalization()

            ])
    def call(self, x, training=None):
        x = self.model(x, training=training)
        return x

class Discriminator(Model):
    def __init__(self, classnumber):
        super(Discriminator, self).__init__()
        self.conv1 = ConvBnRelu(ch = 64, kernel_size=3, strides= 1) # size, 128*128*32
        self.conv2 = ConvBnRelu(ch = 64, kernel_size = 3 , strides = 2) # 64,64,32
        self.conv3 = ConvBnRelu(ch = 64, kernel_size = 3 , strides = 2) # 32,32,16
        self.conv4 = ConvBnRelu(ch = 32, kernel_size = 3 , strides = 1) # 32,32,16
        self.conv5 = ConvBnRelu(ch = 32, kernel_size = 5 , strides = 1) # 16,16,16
        self.conv6 = ConvBnRelu(ch = 16, kernel_size = 5 , strides = 2) # 16,16,16
        self.flatten = layers.Flatten()
        # self.fc1 = layers.Dense(1024)
        # self.bn1 = layers.BatchNormalization()
        self.fc2 = layers.Dense(512, activation = 'linear')
        self.dp1 = layers.Dropout(0.4)
        self.dp2 = layers.Dropout(0.4)
        
        # self.bn2 = layers.BatchNormalization()
        self.out = layers.Dense(classnumber+1)
        
    
        
    def call(self, x, training=False):
        x = tf.reshape(x, [-1, 128, 128, 3])
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x,training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.conv6(x, training=training)
        x = self.flatten(x)
        # x = self.fc1(x)
        # x = tf.nn.relu(x)
        x = self.dp2(x)
        x = self.fc2(x)
        x = self.dp1(x, training = training)
        # x = self.bn2(x, training = training)
        # x= tf.nn.relu(x)
        x = self.out(x)
        # x = tf.nn.relu(self.fc3(x))
        if not training:
            x = tf.nn.softmax(x)
 
        return x
        
class weakDiscriminator(Model):
    def __init__(self, classnumber):
        super(weakDiscriminator, self).__init__()
        self.conv1 = ConvBnRelu(ch = 32, kernel_size=3, strides= 2) # size, 128*128*32
        self.conv2 = ConvBnRelu(ch = 32, kernel_size = 5 , strides = 2) # 64,64,16
        self.conv3 = ConvBnRelu(ch = 16, kernel_size = 3 , strides = 1) # 64,64,16
        self.flatten = layers.Flatten()

        self.fc2 = layers.Dense(512)
        self.bn2 = layers.BatchNormalization()
        self.out = layers.Dense(classnumber+1)
        
    
        
    def call(self, x, training = False):
        x = tf.reshape(x, [-1, 28, 28,1]) ## change the resolution as you like
        x = self.conv1(x, training = training)
        x = self.conv2(x, training = training)   
        x = self.conv3(x, training = training) 
        x = self.flatten(x)
        # x = tf.nn.relu(x)
        x = self.fc2(x)
        # x = self.bn2(x, training = training)
        x= tf.nn.relu(x)
        x =  tf.nn.softmax(self.out(x))
        return x


class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = layers.Dense(4*4*256)
        self.conv1 = ConvTrBnRelu(ch = 256, kernel_size = 3, strides =(2,2)) # size 8,8, 128
        self.conv2 = ConvTrBnRelu(ch = 128, kernel_size = 3, strides =(2,2)) # size 16,16, 64
        self.conv3 = ConvTrBnRelu(ch = 64, kernel_size = 3, strides =(2,2)) # size 32,32, 32
        self.conv4 = ConvTrBnRelu(ch = 32, kernel_size = 3, strides =(2,2)) # size 64,64, 16
        # self.conv4 = layers.Conv2DTranspose(filters = 64, kernel_size = 5,
        #                                     padding= 'same', strides =(1,1)) # size 256, 256, 3
        self.conv5 = ConvTrBnRelu(ch = 32, kernel_size = 5, strides =(2,2)) # size 128,128, 8
        # self.conv5 = layers.Conv2DTranspose(filters = 64, kernel_size = 5,
        #                                     padding= 'same', strides =(1,1)) # size 256, 256, 3
        self.conv6 = layers.Conv2DTranspose(filters = 3, kernel_size = 3,
                                            padding= 'same', strides =(1,1)) # size 256, 256, 3
        
    def call(self, x, training = False):
        x = tf.reshape(x, shape= [-1, 100])
        x = self.fc1(x)
        x =  tf.nn.relu(x)
        x = tf.reshape(x ,[-1, 4,4,256])
        x = self.conv1(x, training =  training)
        x = self.conv2(x, training =  training)
        x = self.conv3(x, training =  training)
        # print(x.shape)
        x = self.conv4(x, training =  training)
        x = self.conv5(x, training =  training)
        # x = self.conv4(x)
        # x = self.conv5(x)
        x = self.conv6(x)
        
        return tf.nn.tanh(x)


class Generator1(Model):
    def __init__(self):
        super(Generator1, self).__init__()
        self.fc1 = layers.Dense(7*7*128)
        self.conv1 = ConvTrBnRelu(ch = 128, kernel_size = 3, strides =(2,2)) # size 8,8, 128
        self.conv2 = ConvTrBnRelu(ch = 64, kernel_size = 3, strides =(1,1)) # size 16,16, 64
        self.conv3 = ConvTrBnRelu(ch = 64, kernel_size = 3, strides =(1,1)) # size 32,32, 32
        self.conv4 = ConvTrBnRelu(ch = 32, kernel_size = 3, strides =(1,1)) # size 64,64, 16
        # self.conv4 = layers.Conv2DTranspose(filters = 64, kernel_size = 5,
        #                                     padding= 'same', strides =(1,1)) # size 256, 256, 3
        self.conv5 = ConvTrBnRelu(ch = 32, kernel_size = 5, strides =(2,2)) # size 128,128, 8
        # self.conv5 = layers.Conv2DTranspose(filters = 64, kernel_size = 5,
        #                                     padding= 'same', strides =(1,1)) # size 256, 256, 3
        self.conv6 = layers.Conv2DTranspose(filters = 1, kernel_size = 3,
                                            padding= 'same', strides =(1,1)) # size 256, 256, 3
        
    def call(self, x, training = False):
        x = tf.reshape(x, shape= [-1, 100])
        x = self.fc1(x)
        x =  tf.nn.relu(x)
        x = tf.reshape(x ,[-1, 7,7,128])
        x = self.conv1(x, training =  training)
        x = self.conv2(x, training =  training)
        x = self.conv3(x, training =  training)
        # print(x.shape)
        x = self.conv4(x, training =  training)
        x = self.conv5(x, training =  training)
        # x = self.conv4(x)
        # x = self.conv5(x)
        x = self.conv6(x)
        
        return tf.nn.tanh(x)