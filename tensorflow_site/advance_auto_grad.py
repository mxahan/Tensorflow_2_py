#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 21:21:34 2020
source : https://www.tensorflow.org/guide/advanced_autodiff
@author: zahid
"""
#%% load library

import  tensorflow as tf 
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize']=(8,6)


#%% controlling gradient 
# gradientTape and stop_recording
x = tf.Variable(2.0) # 2.0 is different from 2!
y = tf.Variable(2.0)

# usual gradient tape to track the gradient
with tf.GradientTape() as t:
    x_sq  = x*x
    
    with t.stop_recording():
        y_sq = y*y
        
    z = x_sq +y_sq
    

grad =  t.gradient(z, {'x':x, 'y':y}) # two variable so store them in dictionary format

print(grad['x'], grad['y'])

#%% reset 

reset = True

with tf.GradientTape() as t:
  y_sq = y * y
  if reset:
    # Throw out all the tape recorded so far
    # which means y
    t.reset()
  z = x * x + y_sq

grad = t.gradient(z, {'x': x, 'y': y})

print('dz/dx:', grad['x'])  # 2*x => 4
print('dz/dy:', grad['y'])

#%% stop gradient
# same thing as previous but this time with stop gradient

with tf.GradientTape() as t:
  y_sq = y**2
  z = x**2 + tf.stop_gradient(y_sq)

grad = t.gradient(z, {'x': x, 'y': y})

print('dz/dx:', grad['x'])  # 2*x => 4
print('dz/dy:', grad['y'])

# Custom gradient later

#%% Multiple tapes

x0 = tf.constant(0.0)
x1 =  tf.constant(0.0)

with tf.GradientTape() as t0, tf.GradientTape() as t1:
    t0.watch(x0)
    t1.watch(x1)
    y0 = tf.math.sin(x0)
    y1 = tf.math.sigmoid(x1)
    
    y = y0 +y1
    
    ys = tf.reduce_sum(y)

# keep all under the with the gradients you want to track

# .gradient is just outside the with blocks

print(t0.gradient(ys, x0).numpy())


print(t1.gradient(ys, x1).numpy())   # sigmoid(x1)*(1-sigmoid(x1)) => 0.25

#%% higher order

x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0

with tf.GradientTape() as t2:
  with tf.GradientTape() as t1:
    y = x * x * x

  # Compute the gradient inside the outer `t2` context manager
  # which means the gradient computation is differentiable as well.
  dy_dx = t1.gradient(y, x)
d2y_dx2 = t2.gradient(dy_dx, x)

print('dy_dx:', dy_dx.numpy())  # 3 * x**2 => 3.0
print('d2y_dx2:', d2y_dx2.numpy())  # 6 * x => 6.0

#%% Advanced stuff

x = tf.random.normal([7,5])
layer = tf.keras.layers.Dense(10, activation=tf.nn.relu)


with tf.GradientTape() as t2:
  # The inner tape only takes the gradient with respect to the input,
  # not the variables.
  with tf.GradientTape(watch_accessed_variables=False) as t1:
    t1.watch(x)
    y = layer(x)
    out = tf.reduce_sum(layer(x)**2)
  # 1. Calculate the input gradient wrt x
  g1 = t1.gradient(out, x)
  # 2. Calculate the magnitude of the input gradient.
  g1_mag = tf.norm(g1)

# 3. Calculate the gradient of the magnitude with respect to the model.
dg1_mag = t2.gradient(g1_mag, layer.trainable_variables) # wrt the trainable variables

    
print([var.shape for var in dg1_mag])

#%% Jacobian 


x = tf.linspace(-10.0, 10.0, 200+1)
delta = tf.Variable(0.0)

with tf.GradientTape() as tape:
  y = tf.nn.sigmoid(x+delta)

dy_dx = tape.jacobian(y, delta)


plt.plot(x.numpy(), y, label='y')
plt.plot(x.numpy(), dy_dx, label='dy/dx')
plt.legend()
_ = plt.xlabel('x')

#%% More breakdown


x = tf.random.normal([7, 5])
layer = tf.keras.layers.Dense(10, activation=tf.nn.relu)

with tf.GradientTape(persistent=True) as tape:
  y = layer(x)

#%%
print(y.shape)

print(layer.kernel.shape)

j = tape.jacobian(y, layer.kernel)

# very simple for each output there is 5,10 output so for 7*10 there is 7*10*5*10 outputs

print(j.shape)

#%%
g = tape.gradient(y, layer.kernel)
print('g.shape:', g.shape)

j_sum = tf.reduce_sum(j, axis=[0, 1])
delta = tf.reduce_max(abs(g - j_sum)).numpy()
assert delta < 1e-3
print('delta:', delta)

#%% Hessian

x = tf.random.normal([7, 5])
layer1 = tf.keras.layers.Dense(8, activation=tf.nn.relu)
layer2 = tf.keras.layers.Dense(6, activation=tf.nn.relu)

with tf.GradientTape() as t2:
  with tf.GradientTape() as t1:
    x = layer1(x)
    x = layer2(x)
    loss = tf.reduce_mean(x**2)

  g = t1.gradient(loss, layer1.kernel)

h = t2.jacobian(g, layer1.kernel)


print(f'layer.kernel.shape: {layer1.kernel.shape}')
print(f'h.shape: {h.shape}')

#%%
n_params = tf.reduce_prod(layer1.kernel.shape)

g_vec = tf.reshape(g, [n_params, 1])
h_mat = tf.reshape(h, [n_params, n_params])

def imshow_zero_center(image, **kwargs):
  lim = tf.reduce_max(abs(image))
  plt.imshow(image, vmin=-lim, vmax=lim, cmap='seismic', **kwargs)
  plt.colorbar()
  
imshow_zero_center(h_mat)
