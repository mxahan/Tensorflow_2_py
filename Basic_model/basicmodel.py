# %% Import libraties
import tensorflow as tf
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import matplotlib.pyplot as plt

import numpy as np
''' Different ways to build a model
1. use subclass model
2. use functional API
3. Use datapipelines
4. Use sequential model
'''
# %% Data load
 
mnist = tf.keras.datasets.mnist
(trX, trY), (teX, teY) =  mnist.load_data()

# %% Use of Dataset.data pipeline

train_x, test_x = tf.cast(trX/255.0, tf.float32), tf.cast(teX/255.0, tf.float32)
train_y, test_y = tf.cast(trY,tf.int64),tf.cast(teY,tf.int64)
epochs=10

#%% ready data
batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices((train_x,
                                                    train_y)).batch(32).shuffle(10000)


train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))


train_dataset = train_dataset.repeat()

test_dataset = tf.data.Dataset.from_tensor_slices((test_x, 
                                                   test_y)).batch(batch_size).shuffle(10000)

test_dataset = train_dataset.repeat()

# %% Sequential model
 
modelsq = tf.keras.models.Sequential([
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(512,activation=tf.nn.relu),
 tf.keras.layers.Dropout(0.2),
 tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

#%% 
steps_per_epoch = len(train_x)//batch_size #required becuase of the repeat() on the dataset

optimiser = tf.keras.optimizers.Adam()

modelsq.compile (optimizer= optimiser,
                loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

# %% 

modelsq.fit(train_dataset, epochs=5, steps_per_epoch = steps_per_epoch)

# %% Subclass Model

class MyModel(tf.keras.Model):
    def __init__(self, num_class = 10):
        super(MyModel, self).__init__()
        
        self.x0 = tf.keras.layers.Flatten()
        self.x1 = tf.keras.layers.Dense(28*28, activation='relu', name = 'd1')
        self.x2 = tf.keras.layers.Dropout(0.2)
        self.preds = tf.keras.layers.Dense(10, activation =  tf.nn.softmax, name = 'd2')
    
    def call(self, inputs):
        
        x = self.x0(inputs)
        x = self.x1(x)
        x = self.x1(x) ## interesting stuff
        x = self.x2(x)
        
        return self.preds(x)

modelsc =  MyModel()

# %% Keras model API
inputs = tf.keras.Input(shape=(28,28))  # Returns a placeholder tensor
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(512, activation='relu',name='d1')(x)
x = tf.keras.layers.Dropout(0.2)(x)
predictions = tf.keras.layers.Dense(10,activation=tf.nn.softmax, name='d2')(x)

modelka = tf.keras.Model(inputs = inputs, outputs =  predictions)

#%% optimizer and fit 

optimiser =  tf.keras.optimizers.Adam()
modelsc.compile(optimizer = optimiser, loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])
with tf.device('/gpu:0'):
    modelsc.fit(trX, trY, batch_size = 32, epochs = 5)


#%% optimize and fit 

modelka.compile(optimizer = optimiser, loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])
with tf.device('/gpu:0'):
    modelka.fit(trX, trY, batch_size = 32, epochs = 5)

#%%

modelsc.evaluate(teX, teY)

modelsc.summary()

modelsq.evaluate(test_dataset,steps=10)

#%% some important point

temp = modelsc.layers # Get all the layers

lay1val = np.array(temp[1].get_weights()) # getting the weights of the the wanted layer.
 
trV = modelsc.trainable_variables # Getting all the trainable variables

#%% Prediction
modelsc.call(np.reshape(teX[0], (1,28,28)))

modelka.predict(np.reshape(teX[0], (1,28,28)))

modelsq.predict(np.reshape(teX[0],[1,28,28]))



