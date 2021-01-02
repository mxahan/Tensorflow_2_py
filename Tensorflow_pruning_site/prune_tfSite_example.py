#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 18:55:54 2020

@author: zahid


https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras'

Modified: Way to prune subclass supported model [it takes a lot of time]


"""
#%% Set up
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import os
from tensorflow.keras import Model, layers
import tempfile

import tensorflow_model_optimization as tfmot
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#%% load data
# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0



#%% define Keras model 
# Define the model architecture

# model = keras.Sequential([
#   keras.layers.InputLayer(input_shape=(28, 28)),
#   keras.layers.Reshape(target_shape=(28, 28, 1)),
#   keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation=tf.nn.relu),
#   keras.layers.MaxPooling2D(pool_size=(2, 2)),
#   keras.layers.Flatten(),
#   keras.layers.Dense(10)
# ])

#%% Alternative model Define

class mnist_model(Model): 
    def __init__(self):
        super(mnist_model, self).__init__()
        self.conv1 = layers.Conv2D(12, kernel_size= (3,3), activation=tf.nn.relu, input_shape = (-1,28,28,1))
        self.maxpool1 =layers.MaxPooling2D(pool_size=(2,2))
        self.flatten = layers.Flatten()
    
    def call(self, x, training=False):
        x= tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.flatten(x)
        return x



model_1 =  mnist_model()

model_2 = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=( 2028,))])

model =  tf.keras.Sequential([model_1,
                              model_2])
model._set_inputs(test_images[0].reshape(-1, 28, 28, 1))


#%%
# Train the digit classification model
def model_compile(model):
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

model =  model_compile(model)

model.fit(
  train_images,
  train_labels,
  epochs=2, 
  validation_data=(test_images, test_labels)
)
#%% training setup
# Alternative
# https://www.tensorflow.org/guide/data
optim = tf.keras.optimizers.Adam(lr = 0.001)

train_data1 = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_data1 = train_data1.repeat().shuffle(buffer_size=100, seed= 8).batch(16).prefetch(1)
scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)

#%% Training
def model_train(model):
    for step, (batch_x, batch_y) in enumerate(train_data1.take(10000), 1): 
        with tf.GradientTape() as g:
            pred =  model(batch_x, training = True) 
            loss =  scce(batch_y, pred) 
        trainable_vars =  model.trainable_variables
        gradients =  g.gradient(loss, trainable_vars)
        optim.apply_gradients(zip(gradients, trainable_vars))
        
        if step % 1000 == 0:
            pred = model(batch_x, training=True)
            # pdb.set_trace()
            loss = scce(batch_y, pred)
            print("step: %i, loss: %f  " % (step, tf.reduce_mean(loss)))

with tf.device('gpu:0'):
    model_train(model)
# the following steps are important to compile the model 

model.compile(optimizer= optim,
              loss=scce,
              metrics=['accuracy'])

#%% Model Evaluation and save model in h5 format

_, baseline_model_accuracy = model.evaluate(
    test_images, test_labels, verbose=0)

new_path =  os.path.join("../../../data/")

tf.keras.models.save_model(model, new_path)

model.save_weights('../../../data/prune')

#%% Actual Pruning begins

import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude


batch_size = 16
epochs = 2
validation_split = 0.1 # 10% of training set will be used for validation set. 

num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

#%% Define model for pruning 

pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}
# model_2 =  tf.keras.Sequential([model_1])
# model_2.load_weights('../../../data/test1')

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
# this didn't work for the normal setup

#%% Model_for_pruning creation [works great.... ]


model_for_pruning =  tf.keras.Sequential(
    [tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(10, input_shape= (2028,)))])
    
model_3 =  tf.keras.Sequential([model_1, model_for_pruning])

model_for_pruning.load_weights('../../../data/prune')

model_for_pruning._set_inputs(test_images[0].reshape(-1, 28, 28, 1))


#%% different try


model_3.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_for_pruning.summary()



#%% Fine tuning 

model_3.fit(train_images, train_labels,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
              callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])

# okay !! now the problem is how to write callbacks in custom training loop!! 

#%% Smaller model creation 

# compressable pruning model

model_for_export =  tfmot.sparsity.keras.strip_pruning(model_for_pruning)

model_4 =  tf.keras.Sequential([model_1, model_for_export])

model_4.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

datt =  np.array([np.random.rand(1, 28, 28, 1)],dtype=np.float32)

model_4(datt)

_, pruned_keras_file = tempfile.mkstemp('.h5')

tf.keras.models.save_model(model_4, pruned_keras_file, include_optimizer=False)

print('Saved pruned Keras model to:', pruned_keras_file)

#%% TfLite Section 


converter = tf.lite.TFLiteConverter.from_keras_model(model_4)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]


pruned_tflite_model = converter.convert()

_, pruned_tflite_file = tempfile.mkstemp('.tflite')

with open(pruned_tflite_file, 'wb') as f:
  f.write(pruned_tflite_model)

print('Saved pruned TFLite model to:', pruned_tflite_file)



#%% Compare model size

def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)


# print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
# print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
print("Size of gzipped pruned TFlite model: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))

