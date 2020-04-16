#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 00:10:10 2020

@author: zahid
"""


import numpy as np
import tensorflow as tf


import os 

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#%% 
VGG_MEAN = [103.939, 116.779, 123.68]

#%%  Truncated version

def vgg_m():       
    input_layer = tf.keras.layers.Input([28,28,1 ])
    
    
    # Block 1
    conv1_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                     use_bias=True, activation='relu', name='conv1_1')(input_layer)
    
    conv1_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                     use_bias=True, activation='relu', name='conv1_2')(conv1_1)
    pool1_1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1_1')
    
    # Block 2
    conv2_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                     use_bias=True, activation='relu', name='conv2_1')(pool1_1)
    conv2_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                     use_bias=True, activation='relu', name='conv2_2')(conv2_1)
    pool2_1 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2_1')
    
    # Block 3
    conv3_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                     use_bias=True, activation='relu', name='conv3_1')(pool2_1)
    conv3_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                     use_bias=True, activation='relu', name='conv3_2')(conv3_1)
    conv3_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                     use_bias=True, activation='relu', name='conv3_3')(conv3_2)
    pool3_1 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3_1')
    
    # Block 4
    conv4_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                     use_bias=True, activation='relu', name='conv4_1')(pool3_1)
    conv4_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                     use_bias=True, activation='relu', name='conv4_2')(conv4_1)
    conv4_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                     use_bias=True, activation='relu', name='conv4_3')(conv4_2)
    pool4_1 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4_1')
    
    # Block 4
    
    flatten = tf.keras.layers.Flatten()(pool4_1)
    fc6 = tf.keras.layers.Dense(units=100, use_bias=True, name='fc6', activation='relu')(flatten)
    fc7 = tf.keras.layers.Dense(units=100, use_bias=True, name='fc7', activation='relu')(fc6)
    fc8 = tf.keras.layers.Dense(units=10, use_bias=True, name='fc8', activation=None)(fc7)
    
    prob = tf.nn.softmax(fc8)
    
    # Build model
    model = tf.keras.Model(input_layer, prob)
    
    return model

#%%
vggM = vgg_m()


#%% Data Load
# MNIST dataset parameters.
num_classes = 10 # total classes (0-9 digits).
num_features = 784 # data features (img shape: 28*28).

# Training parameters.
learning_rate = 0.1
training_steps = 5000
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


#%% Conv Net

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
    x = np.reshape(x, [-1,  28,28,1])
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
            batch_x = np.reshape(batch_x, [-1, 28,28,1])
            pred = neural_net(batch_x)
            loss = cross_entropy_loss(pred, batch_y)
            acc = accr.accuracy(pred, batch_y)
            print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
            print(batch_x.shape)




def conv_net_tr():
    neural_net =  vgg_m()
    
    data_dict = data_ret()

    train_data = data_dict['trD']
    teX, teY  = data_dict['te']
    inarg = (neural_net, train_data)
    train_nn(*inarg)
    
    print(teX.shape)
    print(neural_net(np.reshape(teX[0], [-1,784])))
    
    return neural_net
    

    
#if __name__=="__main__":
#    main()


#%% 
with tf.device('/gpu:0'):
    trCnet = conv_net_tr()

#%%

