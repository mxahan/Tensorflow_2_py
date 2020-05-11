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
import  pdb
import glob
#%% Load Data and prepare data
folds = [x[0] for x in os.walk('../../../Dataset/monkey10K/training/training')]
list.sort(folds)

x=[]
y=[]

re_size = (64,64)

for i,j in enumerate(folds[1:]):
    for imgf in glob.glob(j+'/*.jpg'):
        cvimg = cv2.imread(imgf)
        cvimg = cv2.resize(cvimg, re_size)
        x.append(cvimg)
        y.append(i)
        
x = np.array(x)/255.
y = np.array(y)
    
#%% hyperparameters
lr_generator = 0.0005
lr_discriminator = 0.0005
training_steps = 8000
batch_size = 16
display_step = 100

# Network parameters.
noise_dim = 500 # Noise data points

#%% train test split
xtr, xte, ytr, yte = train_test_split(x,y,test_size = 0.01, random_state=10)

## Add noise for better generator training
xtr = xtr + np.random.normal(0,0.03, xtr.shape)

#%% Data prepare by tensorflow 2

train_data = tf.data.Dataset.from_tensor_slices((xtr, ytr))
train_data = train_data.repeat().shuffle(buffer_size = 16, seed = 5).batch(batch_size).prefetch(1)

#%% GAN Loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_fake):

    ## Alternative 1

    # gen_loss = -tf.math.log(1-disc_fake[:,-1]+10**-10)

    
    ## Alternative 2
    # reform = tf.reshape(disc_fake[:,-1], [batch_size, 1])
    # reform = tf.concat([1-reform, reform], 1)
    
    # gen_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=reform, labels=tf.zeros([batch_size], dtype=tf.int32))
    
    ## Alternative 3
    # gen_loss = (disc_fake[:,-1])**2
    
    ## Alternative  4
    # gen_loss = tf.nn.softmax_cross_entropy_with_logits(
    #     labels = tf.one_hot(tf.ones([batch_size], 'int32')*10, 11), logits = (1-disc_fake))
    
    ## Alternative 5
    
    gen_loss  = disc_fake[:,-1]
    
    return gen_loss



def discriminator_loss(disc_fake, disc_real, real_label):
    # Log likelihood
    # disc_L_sup = tf.nn.softmax_cross_entropy_with_logits(
    #     labels = tf.one_hot(real_label, 11), logits = disc_real)
    disc_L_sup = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = real_label, logits = disc_real)
    
    ## Alternative 1
    # disc_r_un = -tf.math.log(1.0-disc_real[:,-1]+10**-10)
    
    # disc_f_un = -tf.math.log(disc_fake[:,-1]+10**-10)
    
    
    ## Alternative 2 ## not good for generator
    # reform = tf.reshape(disc_fake[:,-1], [batch_size, 1])
    # reform = tf.concat([1-reform, reform], 1)
    
    # disc_f_un = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=reform, labels=tf.ones([batch_size], dtype=tf.int32))
    
    # reform = tf.reshape(disc_real[:,-1], [batch_size, 1])
    # reform = tf.concat([1-reform, reform], 1)
    # disc_r_un = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=reform, labels=tf.zeros([batch_size], dtype=tf.int32))
    
    
    ## Alternative 3    
    # disc_r_un = (disc_real[:,-1])**2
    
    # disc_f_un = (1-disc_fake[:,-1])**2
    
     
    
    ## Alternative  4
    # disc_f_un = tf.nn.softmax_cross_entropy_with_logits(
    #     labels = tf.one_hot(tf.ones([batch_size], 'int32')*10, 11), logits = disc_fake)
    
    # disc_r_un = tf.nn.softmax_cross_entropy_with_logits(
    #     labels = tf.one_hot(tf.ones([batch_size], 'int32')*10, 11), logits = (1-disc_real))


    ## Alternative 5 ()WGAN
    disc_f_un = 1-disc_fake[:,-1]
    
    disc_r_un = disc_real[:,-1]
    
    print(tf.reduce_mean(disc_L_sup).numpy(), 
           tf.reduce_mean(disc_r_un).numpy(), tf.reduce_mean(disc_f_un).numpy())
          
         

    return disc_L_sup + disc_r_un + disc_f_un


#%% Model training + Optimization
optimizer_gen = tf.optimizers.Adam(learning_rate=lr_generator)#, beta_1=0.5, beta_2=0.999)
optimizer_disc = tf.optimizers.Adam(learning_rate=lr_discriminator)#, beta_1=0.5, beta_2=0.999)


def run_optimization(generator, discriminator, real_images, real_label):
    
    # Rescale to [-1, 1], the input range of the discriminator
    real_images = real_images * 2. - 1. # output tanh

    # Generate noise.
    noise = np.random.normal(0, 1., size=[batch_size, noise_dim]).astype(np.float32)
    
    with tf.GradientTape() as g:
            
        fake_images = generator(noise, training=True)
        disc_fake = discriminator(fake_images, training=True)
        disc_real = discriminator(real_images, training=True)
        disc_loss = discriminator_loss(disc_fake, disc_real, real_label)


            
    # Training Variables for each optimizer
    gradients_disc = g.gradient(disc_loss,  discriminator.trainable_variables)
    optimizer_disc.apply_gradients(zip(gradients_disc,  discriminator.trainable_variables))
    
    # use the following only in case of WGAN else comment the for loop
    for lays in discriminator.layers:
        weights = lays.get_weights()
        weights = [np.clip(weight, -0.1, 0.1) for weight in weights]
        lays.set_weights(weights)
    
    # Generate noise.
    for _ in range(2):
        noise = np.random.normal(0., 1., size=[batch_size, noise_dim]).astype(np.float32)
        
        with tf.GradientTape() as g:
                
            fake_images = generator(noise, training=True)
            disc_fake = discriminator(fake_images, training=True)
            gen_loss = generator_loss(disc_fake)   
                
        gradients_gen = g.gradient(gen_loss, generator.trainable_variables)
        optimizer_gen.apply_gradients(zip(gradients_gen, generator.trainable_variables))
        
    return gen_loss, disc_loss



#%% Model loading and main function to train


def ret_GD(generator, discriminator):

    
    for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps + 1)):
    
        if step == 0:
            # Generate noise.
            noise = np.random.normal(-1., 1., size=[batch_size, noise_dim]).astype(np.float32)
            gen_loss = generator_loss(discriminator(generator(noise)))
            disc_loss = discriminator_loss(discriminator(generator(noise)),
                                           discriminator(batch_x),batch_y)
            print("initial: gen_loss: %f, disc_loss: %f" % (tf.reduce_mean(gen_loss), 
                                                            tf.reduce_mean(disc_loss)))
            continue
        
        # Run the optimization.
        args1 =  (generator, discriminator, batch_x, batch_y)
        gen_loss, disc_loss = run_optimization(*args1)
        
        if step % display_step == 0:
            print("step: %i, gen_loss: %f, disc_loss: %f" % (step, tf.reduce_mean(gen_loss), 
                                                            tf.reduce_mean(disc_loss)))
            n = 6
            canvas = np.empty((64 * n, 64 * n,3))
            for i in range(n):
                # Noise input.
                z = np.random.normal(-1., 1., size=[n, noise_dim]).astype(np.float32)
                # Generate image from noise.
                g = generator(z).numpy()
                # Rescale to original [0, 1]
                g = (g + 1.) / 2
                # Reverse colours for better display
                # g = -1 * (g - 1)
                for j in range(n):
                    # Draw the generated digits
                    canvas[i * 64:(i + 1) * 64, j* 64:(j + 1)*64,:] = g[j].reshape([64, 64,3])
            
            plt.figure(figsize=(n, n))
            plt.imshow(canvas, origin="upper", cmap="gray")
            plt.show()

        
    # return generator, discriminator


#%% Model class Definition
from gen_dis_def import Generator, Discriminator, weakDiscriminator
#%% Load model
generator  = Generator()
discriminator =  Discriminator(10)
# wdiscream = weakDiscriminator(10)
#%% train model 

mod_train = (generator, discriminator)
with tf.device('/gpu:0'):
    ret_GD(*mod_train)

#%% Model save 


#%% model test

noise = np.random.normal(0,1, [1,noise_dim])

genimg = generator(noise)

plt.imshow(genimg.numpy()[0])


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
plt.imshow(np.reshape(in2[0,:,:,9].numpy(), [8,8]))

