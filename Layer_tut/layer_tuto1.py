
#%% Same as basic model but functional implementation and breakdown implementation 

#%% Import library

import tensorflow as tf
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import matplotlib.pyplot as plt

from tensorflow.keras import Model, layers

import numpy as np

tf.keras.backend.set_floatx('float32')
#%% Data Load
# MNIST dataset parameters.
num_classes = 10 # total classes (0-9 digits).
num_features = 784 # data features (img shape: 28*28).

# Training parameters.
learning_rate = 0.1
training_steps = 5000
batch_size = 8
display_step = 100

# Network parameters.
n_hidden_1 = 128 # 1st layer number of neurons.
n_hidden_2 = 256 # 2nd layer number of neurons.


def data_ret():
    mnist = tf.keras.datasets.mnist
    # change 1 make it 3d not convenient but lets see
    
    zvec = np.zeros([60000, 28,28,2])
    
    (trX, trY), (teX, teY) =  mnist.load_data()
    
    trX, teX = np.array(trX, np.float32), np.array(teX, np.float32)
    
    trX, teX = trX.reshape([-1, num_features] ), teX.reshape([-1, num_features]) 
    trX, teY =  trX/255., teX/255
        
    trX = trX.reshape(60000, 28, 28, 1)
    
    trX = np.concatenate((trX, zvec), axis = 3)
    
    print(trX.shape)
    train_data = tf.data.Dataset.from_tensor_slices((trX, trY))
    train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
    
    data_dict = {'trD': train_data, 'te' : [teX, teY]}
    
    
    return data_dict

data_ret()
#%% Model Define
baseMod = tf.keras.applications.VGG16(input_shape = [32,32,3],include_top =False)
class NeuralNet(Model):
    def __init__(self, baseMod):
        super(NeuralNet, self).__init__()
        # First fully-connected hidden layer.

   
        self.l1 = baseMod.layers[1]
        self.l2 = baseMod.layers[2]
        self.l3 = baseMod.layers[3]
        
        
        self.flat = layers.Flatten()
        
        self.fc1 =  layers.Dense(10, activation = tf.nn.relu)

    def call(self, x, is_training=False):
        
        x = tf.reshape(x, [-1, 28,28,3])
       
        x = self.l1(x)
        
        x = self.l2(x)
        x = self.l3(x)
        x = self.flat(x)
        x = self.fc1(x)
        
        return x

#%% 

class NeuralNet1(Model):
    def __init__(self):
        super(NeuralNet1, self).__init__()
        # First fully-connected hidden layer.
        
        
        self.flat = layers.Flatten()
        
        self.fc1 =  layers.Dense(10, activation = tf.nn.relu)

    def call(self, x, is_training=False):
        
        x = tf.reshape(x, [-1, 28,28,3])
       

        x = self.flat(x)
        x = self.fc1(x)
        
        return x
#%% Dependency

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
        x = tf.reshape(x,[-1, 28,28,3])
        pred =  neural_net(x)
        loss =  cross_entropy_loss(pred, y)
        
    trainable_variables =  neural_net.trainable_variables
    
    gradients =  g.gradient(loss, trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    
  
#%% train Function
def train_nn(neural_net, train_data):
        
    for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):     
        run_optimization(neural_net, batch_x, batch_y)

        if step % display_step == 0:
            pred = neural_net(batch_x, is_training= 'true')
            loss = cross_entropy_loss(pred, batch_y)
            acc = accr.accuracy(pred, batch_y)
            print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
            print(batch_x.shape)


# def nn_train():
    
#     neural_net = NeuralNet(baseMod)
#     data_dict = data_ret()
#     train_data = data_dict['trD']
#     teX, teY  = data_dict['te']
#     inarg = (neural_net, train_data)
#     train_nn(*inarg)
#     print(teX.shape)
#     print(neural_net(np.reshape(teX[0], [-1,784])))
    
#     return neural_net


#%%


def conv_net_tr():

    neural_net = NeuralNet(baseMod)
    data_dict = data_ret()

    train_data = data_dict['trD']
    teX, teY  = data_dict['te']
    inarg = (neural_net, train_data)
    train_nn(*inarg)
    
    # print(teX.shape)
    # print(neural_net(np.reshape(teX[0], [-1,784])))
    
    return neural_net
    

    
#if __name__=="__main__":
#    main()


#%%  run the code
with tf.device('/gpu:0'):
    trCnet = conv_net_tr()


#%%
data_all =  data_ret()
teX, _ =  data_all['te']

#%% Delete the whole GPU memory without the kernel repeater
from numba import cuda
cuda.select_device(0)



    
cuda.close()