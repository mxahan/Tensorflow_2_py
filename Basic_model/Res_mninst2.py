#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:22:38 2020
"""

import os 
os.environ["TF_ALLOW_GPU_GROWTH"]='true'

import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from tensorflow.keras import  layers, Model
#%%

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')




(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32)/255., x_test.astype(np.float32)/255.
# [b, 28, 28] => [b, 28, 28, 1]
x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)
# one hot encode the labels. convert back to numpy as we cannot use a combination of numpy
# and tensors as input to keras
y_train_ohe = tf.one_hot(y_train, depth=10).numpy()
y_test_ohe = tf.one_hot(y_test, depth=10).numpy()



print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
#%% Resnet In this case initial filter then the resnet block


def conv3_3(ch, stride = 1, kernel  = (3,3)):
    return layers.Conv2D(ch, kernel, strides = stride, padding = "same", 
                         use_bias= False, kernel_initializer = tf.random_normal_initializer())


class resnetBlock(Model):
    def __init__(self, ch, strides= 1, res_path = False):
        super(resnetBlock, self).__init__()
        
        self.ch =  ch
        self.strides = strides
        self.res_path = res_path
        
        self.conv1 = conv3_3(ch, strides)
        self.bn1 = layers.BatchNormalization()


        self.conv2 = conv3_3(ch)       
        self.bn2 = layers.BatchNormalization()
        
        if res_path:
            self.dcv = conv3_3(ch, strides, kernel  = (1,1))
            self.dbn =  layers.BatchNormalization()
            
    def call(self, inputs, training = None):
        reside =  inputs
        x = tf.nn.relu(self.bn1(inputs, training = training))
        x = self.conv1(x)
        
        
        
        x= tf.nn.relu(self.bn2(x, training= training))
        x= self.conv2(x)
        
        if self.res_path:
            reside = self.dbn(inputs, training = training)
            reside =  tf.nn.relu(reside)
            reside =  self.dcv(reside)
            
        x = x +reside
        
        return x
    


class ResNet(keras.Model):

    def __init__(self, block_list, num_classes, initial_filters=16, **kwargs):
        super(ResNet, self).__init__(**kwargs)

        self.num_blocks = len(block_list)
        self.block_list = block_list

        self.in_channels = initial_filters
        self.out_channels = initial_filters
        self.conv_initial = conv3_3(self.out_channels)

        self.blocks = keras.models.Sequential(name='dynamic-blocks')

        # build all the blocks
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):

                if block_id != 0 and layer_id == 0:
                    block = resnetBlock(self.out_channels, strides=2, res_path=True)
                else:
                    if self.in_channels != self.out_channels:
                        residual_path = True
                    else:
                        residual_path = False
                    block = resnetBlock(self.out_channels, res_path=residual_path)

                self.in_channels = self.out_channels

                self.blocks.add(block)

            self.out_channels *= 2

        self.final_bn = keras.layers.BatchNormalization()
        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(num_classes)
        
    def call(self, inputs, training=None):

        out = self.conv_initial(inputs)

        out = self.blocks(out, training=training)

        out = self.final_bn(out, training=training)
        out = tf.nn.relu(out)

        out = self.avg_pool(out)
        out = self.fc(out)


        return out

#%% same old training network
        
        
def main():
    num_classes = 10
    batch_size = 32
    epochs = 1

    # build model and optimizer
    model = ResNet([2, 2, 2], num_classes)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.build(input_shape=(None, 28, 28, 1))
    print("Number of variables in the model :", len(model.variables))
    model.summary()

    # train
    model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test_ohe), verbose=1)

    # evaluate on test set
    scores = model.evaluate(x_test, y_test_ohe, batch_size, verbose=1)
    print("Final test loss and accuracy :", scores)




if __name__ == '__main__':
    main()
        

