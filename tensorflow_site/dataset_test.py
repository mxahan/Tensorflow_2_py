#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 01:13:02 2020

@author: zahid
"""

import tensorflow as tf
import numpy as np

import os
os.environ["TF_FORCE_ALLOW_GPU_GROWTH"]='true'

#%% dataset define

dataset =  tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6])

dataset =  dataset.map(lambda x: x*2-1)

print(list(dataset.as_numpy_iterator()))


dataset = dataset.shuffle(2).batch(2)

for x in dataset:
    print(x)

#%% 

x = np.array([1,2,3])
y = np.array([[1,2],[2,3],[3,4]])

dataset =  tf.data.Dataset.from_tensor_slices((x, y)) 

def change_x(x, y):
    return  x**2, y

dataset = dataset.map(lambda x,y: change_x(x,y)) # can use any function


print(list(dataset.as_numpy_iterator()))


#%% Load and transform

x = np.array([1,2,3])

dataset =  tf.data.Dataset.from_tensor_slices((x)) # 2 braces doesn't matter

# dataset = tf.data.Dataset.range(10)  # another way to define 

def any_func(x):
    return x*2-1


dataset = dataset.map(lambda x: x**2) # can use any function following line 

#dataset = dataset.map(lambda x: any_func(x))


print(list(dataset.as_numpy_iterator()))

#%% Methods

'''
apply(function)

as_numpy_iterator()

batch

cache

concatenate

enumerate 

filter

flat_map  %% self explanatory

from_generator %% From iteration tool 

from_tensor_slices %% prepare dataset for training

from_tensors

interleave

list_files

map %% very important

options

padded_batch

prefetch

range

reduce

repeat %% repeat the same dataset

shard %% sampling 

shuffle 

skip

take %% how many do you want to take

zip  %% combine dataset

... from generator

https://www.tensorflow.org/api_docs/python/tf/data/Dataset
'''
# can use apply method

dataset =  tf.data.Dataset.range(10)

def dataset_func(ds):
    return ds.filter( lambda x: x%2 ==0)  # use any selection function

def dataset_func1(ds):
    return ds.filter( lambda x: x%2 ==1) 

dataset = dataset.apply(dataset_func)

dataset = dataset.batch(3)

print(list(dataset.as_numpy_iterator()))

a =  tf.data.Dataset.range(10)
b =  tf.data.Dataset.range(10)
#
a = a.apply(dataset_func)
b = b.apply(dataset_func1)

ds = a.concatenate(b)
#
a = a.enumerate(start=3)

for element in a.as_numpy_iterator():
    print(element)

#

b = b.filter(lambda x: x<5)

print(list(b.as_numpy_iterator())) # similar as apply  but not user define funcion
# apply itself use filter inside 

def filt_fn(x):
    return tf.math.equal(x,3)

b = b.filter(filt_fn)

print(list(b.as_numpy_iterator()))

#

import itertools

def gen():
    for i in itertools.count(1):
        yield (i, [1]*i)

dataset = tf.data.Dataset.from_generator(gen, (tf.int64, tf.int64),
                                         (tf.TensorShape([]),tf.TensorShape([None])))

print(list(dataset.take(3).as_numpy_iterator()))
