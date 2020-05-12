#%% Load library
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow import keras
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import glob
#%% Load Data and prepare data
folds = [x[0] for x in os.walk('../../../Dataset/monkey10K/training/training')]
list.sort(folds)

x=[]
y=[]

re_size = (32,32)

for i,j in enumerate(folds[1:]):
    for imgf in glob.glob(j+'/*.jpg'):
        cvimg = cv2.imread(imgf)
        cvimg = cv2.resize(cvimg, re_size)
        x.append(cvimg)
        y.append(i)
        cvimg = cv2.rotate(cvimg,cv2.ROTATE_90_CLOCKWISE)
        x.append(cvimg)
        y.append(i)
        cvimg = cv2.rotate(cvimg,cv2.ROTATE_90_COUNTERCLOCKWISE)
        x.append(cvimg)
        y.append(i)
        
x = np.array(x)/255.
y = np.array(y)
    
#%% hyperparameters
lr_generator = 0.00005
lr_discriminator = 0.01
training_steps = 3000
batch_size = 16
display_step = 100

# Network parameters.
noise_dim = 100 # Noise data points

#%% train test split
xtr, xte, ytr, yte = train_test_split(x,y,test_size = 0.01, random_state=42)

#%% Data prepare by tensorflow 2

train_data = tf.data.Dataset.from_tensor_slices((xtr, ytr))
train_data = train_data.repeat().shuffle(buffer_size = 16, seed = 3).batch(batch_size).prefetch(1)

#%% GAN Loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def generator_loss(disc_fake):
    gen_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=disc_fake, labels=tf.ones([batch_size], dtype=tf.int32))
    # gen_loss = cross_entropy(tf.ones_like(disc_fake), disc_fake)
    
    return gen_loss

def discriminator_loss(disc_fake, disc_real):
    disc_loss_real = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=disc_real, labels=tf.ones([batch_size], dtype=tf.int32))
    disc_loss_fake = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=disc_fake, labels=tf.zeros([batch_size], dtype=tf.int32))
    # disc_loss_real = cross_entropy(tf.ones_like(disc_real), disc_real)
    # disc_loss_fake = cross_entropy(tf.zeros_like(disc_fake), disc_fake)
    return disc_loss_real + disc_loss_fake

#%% Model training + Optimization
optimizer_gen = tf.optimizers.Adam(learning_rate=lr_generator)#, beta_1=0.5, beta_2=0.999)
optimizer_disc = tf.optimizers.Adam(learning_rate=lr_discriminator)#, beta_1=0.5, beta_2=0.999)

import pdb

def run_optimization(generator, discriminator, real_images):
    
    # Rescale to [-1, 1], the input range of the discriminator
    real_images = real_images * 2. - 1. # output tanh

    # Generate noise.
    noise = np.random.normal(-1., 1., size=[batch_size, noise_dim]).astype(np.float32)
    
    with tf.GradientTape() as g1, tf.GradientTape() as g2:
        fake_images = generator(noise, training=True)
        disc_fake = discriminator(fake_images, training=True)
        disc_real = discriminator(real_images, training=True)
        disc_loss = discriminator_loss(disc_fake, disc_real)
        gen_loss = generator_loss(disc_fake)   
            
    # Training Variables for each optimizer
    gradients_disc = g1.gradient(disc_loss,  discriminator.trainable_variables)
    optimizer_disc.apply_gradients(zip(gradients_disc,  discriminator.trainable_variables))
    
    gradients_gen = g2.gradient(gen_loss, generator.trainable_variables)
    optimizer_gen.apply_gradients(zip(gradients_gen, generator.trainable_variables))
        
    # Generate noise.
    for i in range(1):
        noise = np.random.normal(-1., 1., size=[batch_size, noise_dim]).astype(np.float32)
        
        with tf.GradientTape() as g:
                
            fake_images = generator(noise, training=True)
            disc_fake = discriminator(fake_images, training=True)
            gen_loss = generator_loss(disc_fake)   
                
        gradients_gen = g.gradient(gen_loss, generator.trainable_variables)
        optimizer_gen.apply_gradients(zip(gradients_gen, generator.trainable_variables))
        
    return gen_loss, disc_loss



#%% Model loading and main function to train

def ret_GD(generator, discriminator):

    
    for step, (batch_x, _) in enumerate(train_data.take(training_steps + 1)):
    
        if step == 0:
            # Generate noise.
            noise = np.random.normal(-1., 1., size=[batch_size, noise_dim]).astype(np.float32)
            gen_loss = generator_loss(discriminator(generator(noise)))
            # pdb.set_trace()
            disc_loss = discriminator_loss(discriminator(batch_x), discriminator(generator(noise)))
            print("initial: gen_loss: %f, disc_loss: %f" % (tf.reduce_mean(gen_loss),
                                                            tf.reduce_mean(disc_loss)))
            
        
            continue
        
        # Run the optimization.
        args1 =  (generator, discriminator, batch_x)
        gen_loss, disc_loss = run_optimization(*args1)
        
        if step % display_step == 0:
            print("step: %i, gen_loss: %f, disc_loss: %f" % (step, tf.reduce_mean(gen_loss), 
                                                            tf.reduce_mean(disc_loss)))
            n = 6
            canvas = np.empty((re_size[0] * n, re_size[0] * n,3))
            for i in range(n):
                # Noise input.
                z = np.random.normal(0., 1., size=[n, noise_dim]).astype(np.float32)
                # Generate image from noise.
                g = generator(z).numpy()
                # Rescale to original [0, 1]
                g = (g + 1.) / 2
                # Reverse colours for better display
                # g = -1 * (g - 1)
                for j in range(n):
                    # Draw the generated digits
                    canvas[i * re_size[0]:(i + 1) * re_size[0], j* re_size[0]
                           :(j + 1)*re_size[0],:] = g[j].reshape([re_size[0],re_size[0],3])
            
            plt.figure(figsize=(n, n))
            plt.imshow(canvas, origin="upper", cmap="gray")
            plt.show()



#%% Model class Definition
from gen_dis_def import Generator, Discriminator
#%% Load model
generator  = Generator()
discriminator =  Discriminator(1)
#%% train model 
mod_train = (generator, discriminator)
with tf.device('/gpu:0'):
    ret_GD(*mod_train)

#%% Model save 


#%% model test

noise = np.random.normal(-1,1, [1,noise_dim]).astype(np.float32)

genimg = generator(noise)

plt.imshow(0.5*(genimg.numpy()[0]+1))


#%%
# Generate images from noise, using the generator network.
n = 6
canvas = np.empty((256 * n, 256 * n,3))
for i in range(n):
    # Noise input.
    z = np.random.normal(-1., 1., size=[n, noise_dim]).astype(np.float32)
    # Generate image from noise.
    g = generator(z).numpy()
    # Rescale to original [0, 1]
    g = (g + 1.) / 2
    # Reverse colours for better display
    g = -1 * (g - 1)
    for j in range(n):
        # Draw the generated digits
        canvas[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256,:] = g[j].reshape([256, 256,3])

plt.figure(figsize=(n, n))
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()


#%% Seeing inside the layers of genenrator

l1 = generator.layers[0]
in1 = l1(noise)
l2 = generator.layers[1]
in2 = l2(tf.reshape(in1, [-1,4,4,512]))
plt.imshow(np.reshape(in2[0,:,:,10].numpy(), [8,8]))

