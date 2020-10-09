#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 22:00:35 2020

@author: zahid
"""
''' This is a method of post training quantization 

https://www.tensorflow.org/lite/performance/post_training_quantization

'''


#%% Dynamic range quantization 
'''Save the model as .pb file with variables and thea assets folder associated
Now to choose option for the fp or integer limit for the task. 
direction = saved_model_dir 

'''
## Default does the interger 8 bit quantization 
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

#%% Advanced 

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]


def representative_dataset_gen():
  for _ in range(num_calibration_steps):
    # Get sample input data as a numpy array in a method of your choosing.
    yield [input]

converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()

## coral edge TPU 

import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_dataset_gen():
  for _ in range(num_calibration_steps):
    # Get sample input data as a numpy array in a method of your choosing.
    yield [input]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()

#%% float 16
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()

# More advanced option in the original website


#%% inference is the most important


# Linux option because why not!
# load and run in python linux

# must have the file stored in the .tflite format

import numpy as np
import tensorflow as tf


# load model and allocate tensor

interpreter =  tf.lite.Interpreter(model_path="***.tflite")
interpreter.allocate_tensors()


# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)


# for a detailed example please follow my other example in project_rppg
# may be public 
# I am planning to perform a example in this section later 

# source : https://www.tensorflow.org/lite/guide/inference#linux
# talked about linux based device : https://www.tensorflow.org/lite/guide/python

