#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:15:42 2020

@author: zahid
"""

import tensorflow as tf
import numpy as np
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#%% Eagar mode

a = np.array([1.,2.])

b = np.array([2.,2.])

tf.add(a,b)

#%% Auto Graph

@tf.function #Key line for graph 
def add_fc(a,b):
    return tf.add(a,b)

print(add_fc(a,b))

#%%

def add_fc(a, b):
    return tf.add(a, b)

print(tf.autograph.to_code(add_fc))

#%% Model definition

# Flatten
model = tf.keras.models.Sequential()
# Add layers
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

#%% will output and no differentiation
model_output = model.predict(np.zeros((1, 30)))
model_output

#%%

model_output = model(np.zeros((1, 30)))
model_output
#%%
@tf.function
def predict(x):
    return model(x)

model_output = predict(np.zeros((1, 30)))
print(model_output)

#%% Fashion Dataset 
fashion_mnist = tf.keras.datasets.fashion_mnist
(images, targets), (images_test, targets_test) = fashion_mnist.load_data()
#%%