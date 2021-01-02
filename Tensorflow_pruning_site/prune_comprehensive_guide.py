#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 20:49:23 2020

@author: zahid
"""
#  Source: https://www.tensorflow.org/model_optimization/guide/pruning/comprehensive_guide

#%%   Set Up 
 
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import os
from tensorflow.keras import Model, layers
import tempfile
import tensorflow_model_optimization as tfmot
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#%% Preparing some function 

input_shape = [20]
x_train = np.random.randn(1, 20).astype(np.float32)
y_train = tf.keras.utils.to_categorical(np.random.randn(1), num_classes=20)


def setup_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(20, input_shape=input_shape),
      tf.keras.layers.Flatten()
  ])
  return model

def setup_pretrained_weights():
  model = setup_model()

  model.compile(
      loss=tf.keras.losses.categorical_crossentropy,
      optimizer='adam',
      metrics=['accuracy']
  )
  model.fit(x_train, y_train)

  _, pretrained_weights = tempfile.mkstemp('.tf')

  model.save_weights(pretrained_weights)

  return pretrained_weights

def get_gzipped_model_size(model):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, keras_file = tempfile.mkstemp('.h5')
  model.save(keras_file, include_optimizer=False)

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(keras_file)

  return os.path.getsize(zipped_file)

setup_model()
pretrained_weights = setup_pretrained_weights() # directory of the save model 

#%% Model define 

base_model = setup_model()
base_model.load_weights(pretrained_weights)


model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)

model_for_pruning.summary()

#%% Prune layers for functional and sequential model 

base_model =  setup_model()
base_model.load_weights(setup_pretrained_weights())


def apply_pruning_to_dense(layer):
  if isinstance(layer, tf.keras.layers.Dense):
    return tfmot.sparsity.keras.prune_low_magnitude(layer)
  return layer

model_for_pruning = tf.keras.models.clone_model(
    base_model,
    clone_function=apply_pruning_to_dense,
)

model_for_pruning.summary()

#%% Functional example

# Use `prune_low_magnitude` to make the `Dense` layer train with pruning.
i = tf.keras.Input(shape=(20,))
x = tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(10))(i)
o = tf.keras.layers.Flatten()(x)
model_for_pruning = tf.keras.Model(inputs=i, outputs=o)

model_for_pruning.summary()

#%% Sequential Model

# Use `prune_low_magnitude` to make the `Dense` layer train with pruning.
model_for_pruning = tf.keras.Sequential([
  tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(20, input_shape=input_shape)),
  tf.keras.layers.Flatten()
])

model_for_pruning.summary()

#%%

class MyDenseLayer(tf.keras.layers.Dense, tfmot.sparsity.keras.PrunableLayer):

  def get_prunable_weights(self):
    # Prune bias also, though that usually harms model accuracy too much.
    return [self.kernel, self.bias]

# Use `prune_low_magnitude` to make the `MyDenseLayer` layer train with pruning.
model_for_pruning = tf.keras.Sequential([
  tfmot.sparsity.keras.prune_low_magnitude(MyDenseLayer(20, input_shape=input_shape)),
  tf.keras.layers.Flatten()
])

model_for_pruning.summary()


#%% Model training

# Define the model.
base_model = setup_model()
base_model.load_weights(pretrained_weights) # optional but recommended for model accuracy
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)

log_dir = tempfile.mkdtemp()
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    # Log sparsity and other metrics in Tensorboard.
    tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
]

model_for_pruning.compile(
      loss=tf.keras.losses.categorical_crossentropy,
      optimizer='adam',
      metrics=['accuracy']
)

model_for_pruning.fit(
    x_train,
    y_train,
    callbacks=callbacks,
    epochs=2,
)


#%% Custom training Loop
