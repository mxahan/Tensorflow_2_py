#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:03:57 2020

@author: zahid
"""


from __future__ import absolute_import, division, print_function

import numpy as np
import random
import requests
import string
import tarfile
import tensorflow as tf

#%%

evens =  np.arange(0,100, step = 2, dtype = np.int32)
evens_l =  np.zeros(50, dtype = np.int32)

odds =  np.arange(1,100, step = 2, dtype = np.int32)
odds_l =  np.ones(50, dtype = np.int32)


feats =  np.concatenate([evens, odds])
labs =  np.concatenate([evens_l,odds_l])
#%%

data =  tf.data.Dataset.from_tensor_slices((feats, labs))

data =  data.repeat()

data =  data.shuffle(buffer_size = 50)

data = data.batch(batch_size=5)

data = data.prefetch(buffer_size= 1)

#%%

for bat_X, bat_Y in data.take(10):
    print(bat_X, bat_Y)
    
    
#%%