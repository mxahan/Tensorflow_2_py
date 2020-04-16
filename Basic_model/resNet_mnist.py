#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:09:05 2020

@author: zahid
"""

import tensorflow as tf

from tensorflow.keras import layers
import numpy as np
#%%

class BasicBlock(tf.keras.Model):
    expansion = 1
    
    def __init__(self, in_ch, out_ch, strides = 1):
        super(BasicBlock, self).__init__()
        
        self.conv1 =  tf.keras.layers.Conv2D(out_ch, kernel_size = 3, 
                                             strides = strides, 
                                             padding = "same", use_bias= False)
        self.bn1 =  layers.BatchNormalization()
        
        self.conv2 = layers.Conv2D(out_ch, kernel_size = 3 , strides = 1, 
                                   padding ="same", use_bias=False)
        
        self.bn2 = layers.BatchNormalization()
        
        if strides !=1 or in_ch !=self.expansion*out_ch:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_ch, kernel_size = 1, 
                              strides =strides, use_bias=False), 
                layers.BatchNormalization()])
        else: 
            self.shortcut =  lambda x, _ :x 
    
    def call(self, x, training =False):
        out =  tf.nn.relu(self.bn1(self.conv1(x), training= training))
        
        out =  self.bn2(self.conv2(out), training= training)
        
        out += self.shortcut(x, training)
        
        return tf.nn.relu(out)
    
    
    
class Bottlenect(tf.keras.Model):
    expansion  = 4
    
    def __init__(self, in_ch, out_ch, strides = 1):
        super(Bottlenect, self).__init__()
        
        self.conv1 = tf.keras.Sequential([  layers.Conv2D(out_ch, 1,1, use_bias = False)
                     , tf.keras.layers.BatchNormalization()])
        
        self.conv2 =  layers.Conv2D(out_ch, 3, strides, padding ="same", use_bias = False)
        
        self.bn2 = layers.BatchNormalization()
        
        self.conv3 =  layers.Conv2D(out_ch*self.expansion, 1, 1, use_bias = False)
        
        self.bn3  = layers.BatchNormalization()
        
        if strides!=1 or in_ch!=self.expansion*out_ch:
            
            self.shortcut =  tf.keras.Sequential([
                layers(self.expansion*out_ch, kernel_size = 1, strides= strides, use_bias=False),
                layers.BatchNormalization()])
        else:
            self.shortcut = lambda x,_: x
            
    def call(self, x, training=False):
        
        out = tf.nn.relu(self.conv1(x, training))
        out = tf.nn.relu(self.bn2(self.conv2(x), training))
        out = self.bn3(self.conv3(out), training)
        
        out += self.shortcut(x, training)
        
        return tf.nn.relu(out)
    
#%%

class ResNet(tf.keras.Model):
    def __init__(self, block, num_block, num_classes = 10):
        
        super(ResNet, self).__init__()
        
        self.in_ch = 64
        self.conv1 =  layers.Conv2D(64, 3 ,1 , padding= "same", use_bias =False)
        self.bn1 = layers.BatchNormalization()
        
        self.layer1 = self._make_layer(block, 64, num_block[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, num_block[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_block[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_block[3], stride = 2)
        
        self.avg_pool2d = tf.keras.layers.AveragePooling2D(4)
        self.linear = tf.keras.layers.Dense(units=num_classes, activation="softmax")

        
    def _make_layer(self, block, out_ch, num_block, stride):
        strides = [stride]+[1]*(num_block-1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_ch, out_ch, stride))
            self.in_ch =out_ch*block.expansion
        return tf.keras.Sequential(layers)
    
    
    def call (self, x, training= False):
        x = tf.reshape(x, [-1, 28, 28, 1])
        out = tf.nn.relu(self.bn1(self.conv1(x), training))
        
        out = self.layer1(out, training=training)
        out = self.layer2(out, training=training)
        out = self.layer3(out, training=training)
        out = self.layer4(out, training=training)
        
        out = self.avg_pool2d(out)
        out = tf.reshape(out, (out.shape[0], -1))
        out = self.linear(out)
        
        return  out

#%%

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])
        

#%%


#%% Data Load
# MNIST dataset parameters.
num_classes = 10 # total classes (0-9 digits).
num_features = 784 # data features (img shape: 28*28).

# Training parameters.
learning_rate = 0.1
training_steps = 1000
batch_size = 16
display_step = 100

# Network parameters.
n_hidden_1 = 128 # 1st layer number of neurons.
n_hidden_2 = 256 # 2nd layer number of neurons.


def data_ret():
    mnist = tf.keras.datasets.mnist
    (trX, trY), (teX, teY) =  mnist.load_data()
    
    trX, teX = np.array(trX, np.float32), np.array(teX, np.float32)
    
    trX, teX = trX.reshape([-1, num_features] ), teX.reshape([-1, num_features]) 
    trX, teY =  trX/255., teX/255
        
    train_data = tf.data.Dataset.from_tensor_slices((trX, trY))
    train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
    
    data_dict = {'trD': train_data, 'te' : [teX, teY]}
    
    return data_dict


#%%

def cross_entropy_loss(x, y):
    y =  tf.cast(y, tf.int64)
    loss  =  tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = x)
    
    return tf.reduce_mean(loss)

class accr():
    def accuracy(y_pred, y_true):    
        cor_pred = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))    
        return tf.reduce_mean(tf.cast(cor_pred, tf.float32), axis = -1)



optimizer = tf.optimizers.SGD(learning_rate)

#%% Run Optimizer

def run_optimization(neural_net, x,y):    
    with tf.GradientTape() as g:
        pred =  neural_net(x, training = True)
        loss =  cross_entropy_loss(pred, y)
        
    trainable_variables =  neural_net.trainable_variables
    
    gradients =  g.gradient(loss, trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    
  
#%%
def train_nn(neural_net, train_data):
        
    for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):     
        run_optimization(neural_net, batch_x, batch_y)

        if step % display_step == 0:
            pred = neural_net(batch_x, training=True)
            loss = cross_entropy_loss(pred, batch_y)
            acc = accr.accuracy(pred, batch_y)
            print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
            print(batch_x.shape)


def nn_train():
    
    neural_net = ResNet18()
    data_dict = data_ret()
    train_data = data_dict['trD']
    teX, teY  = data_dict['te']
    inarg = (neural_net, train_data)
    train_nn(*inarg)
    print(teX.shape)
    print(neural_net(np.reshape(teX[0], [-1,784])))
    
    return neural_net

trM =nn_train()

#%% 