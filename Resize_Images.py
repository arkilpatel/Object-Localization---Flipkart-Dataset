#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,Lambda
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D,UpSampling2D,Concatenate,Conv2DTranspose
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
#import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.callbacks import LearningRateScheduler,Callback
from keras.optimizers import RMSprop
import glob
import cv2
import os
from math import *
from keras.applications.vgg16 import VGG16 
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import h5py
from keras.utils import to_categorical
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from keras.preprocessing.image import ImageDataGenerator
from skimage import measure
from skimage.transform import rotate
import tensorflow as tf
#from imutils import contours
#import imutils
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df_train = pd.read_csv("training.csv")
df_test = pd.read_csv("test.csv")


# In[3]:


df_train.head()


# In[4]:


image_names = df_train['image_name']


# In[21]:


cols = ['x1','x2','y1','y2']
cords = df_train[cols]


# In[6]:

imgs = []
for i in range(len(image_names)):
    print(i)
    path = '/media/vader13/DATA/Images/' + image_names[i]
    img = cv2.imread(path)
    img = np.asarray(img)
    img = cv2.resize(img, (128, 96)) 
    imgs.append(img)


# In[8]:


print(imgs[0].shape)


# In[9]:


import pickle

with open("output.bin", "wb") as output:
    pickle.dump(imgs, output)

