#%% Same as basic model but functional implementation and breakdown implementation 

#%% Import library

import tensorflow as tf
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import matplotlib.pyplot as plt

from tensorflow.keras import Model, layers

import numpy as np

#%% Data Load
# MNIST dataset parameters.
num_classes = 10 # total classes (0-9 digits).
num_features = 784 # data features (img shape: 28*28).

# Training parameters.
learning_rate = 0.1
training_steps = 5000
batch_size = 256
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

class NeuralNet(Model):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # First fully-connected hidden layer.
        self.fc1 = layers.Dense(n_hidden_1, activation=tf.nn.relu)
        self.fc2 = layers.Dense(n_hidden_2, activation=tf.nn.relu)
        self.out = layers.Dense(num_classes, activation=tf.nn.softmax)

    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x

# Build neural network model.
#%% Conv Net
class ConvNet(Model):
    # Set layers.
    def __init__(self):
        super(ConvNet, self).__init__()
        # Convolution Layer with 32 filters and a kernel size of 5.
        self.conv1 = layers.Conv2D(32, kernel_size=5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool1 = layers.MaxPool2D(2, strides=2)

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool2 = layers.MaxPool2D(2, strides=2)

        # Flatten the data to a 1-D vector for the fully connected layer.
        self.flatten = layers.Flatten()

        # Fully connected layer.
        self.fc1 = layers.Dense(1024)
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout = layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    # Set forward pass.
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=is_training)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x


#%%

def cross_entropy_loss(x, y):
    y =  tf.cast(y, tf.int64)
    loss  =  tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = x)
    
    return tf.reduce_mean(loss)


def accuracy(y_pred, y_true):    
    cor_pred = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))    
    return tf.reduce_mean(tf.cast(cor_pred, tf.float32), axis = -1)



optimizer = tf.optimizers.SGD(learning_rate)

#%% Run Optimizer

def run_optimization(neural_net, x,y):    
    with tf.GradientTape() as g:
        pred =  neural_net(x, is_training = True)
        loss =  cross_entropy_loss(pred, y)
        
    trainable_variables =  neural_net.trainable_variables
    
    gradients =  g.gradient(loss, trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    
  
#%%
def train_nn(neural_net, train_data):
        
    for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):     
        run_optimization(neural_net, batch_x, batch_y)
        
        if step % display_step == 0:
            pred = neural_net(batch_x, is_training=True)
            loss = cross_entropy_loss(pred, batch_y)
            acc = accuracy(pred, batch_y)
            print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))


def nn_train():
    
    neural_net = NeuralNet()
    data_dict = data_ret()
    train_data = data_dict['trD']
    teX, teY  = data_dict['te']
    inarg = (neural_net, train_data)
    train_nn(*inarg)
    print(teX.shape)
    print(neural_net(np.reshape(teX[0], [-1,784])))
    
    return neural_net

def conv_net_tr():
    neural_net =  ConvNet()
    
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
tr_net =  nn_train()

#%% 
with tf.device('/gpu:0'):
    trCnet = conv_net_tr()
#%%

data_all =  data_ret()

#%%

teX, _ =  data_all['te']