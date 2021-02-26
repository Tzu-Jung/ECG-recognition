# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

#import function
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, SpatialDropout2D, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop
from keras.callbacks import EarlyStopping
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.engine import training
from typing import Tuple
import time
from keras.layers.normalization import BatchNormalization

print(tf.__version__)


#parameter setting
batch_size = 32
img_height = 100
img_width = 100

#training data's parameter setting
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "/home/muffin/CNN_zip/CNN/CNNData/second/Data/",
  label_mode='categorical',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#validation parameter setting
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "/home/muffin/CNN_zip/CNN/CNNData/second/Data/",
  label_mode='categorical',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


class_names = train_ds.class_names
print(class_names)


# set the parameter of optimizer
optimizer = RMSprop(lr=1e-4)


EStop = EarlyStopping(monitor='val_loss', min_delta=0, 
                    patience=10, verbose=1, mode='auto')


from keras.applications.resnet50 import ResNet50
num_classes = 5
input_shape = (img_height, img_width, 3)
model_resnet = ResNet50(include_top=True, weights=None,
                  input_shape=input_shape, classes=num_classes)
model_resnet.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
model_resnet.summary()
  
  
  
  
# training
History = model_resnet.fit(
  train_ds,
  validation_data=val_ds,
  epochs=300,
  callbacks=[EStop]
)


#save model
import time
timestr = time.strftime("%Y%m%d_%H%M%S")
model_resnet.save('resnet_model_{}.h5'.format(timestr)) 

