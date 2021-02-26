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


# alexnet model
model_alexnet = Sequential()
model_alexnet.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(100,100,3),padding='valid',activation='relu',kernel_initializer='uniform'))
model_alexnet.add(BatchNormalization())

model_alexnet.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model_alexnet.add(BatchNormalization())

model_alexnet.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model_alexnet.add(BatchNormalization())

model_alexnet.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model_alexnet.add(BatchNormalization())

model_alexnet.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model_alexnet.add(BatchNormalization())

model_alexnet.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model_alexnet.add(BatchNormalization())

model_alexnet.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model_alexnet.add(BatchNormalization())

model_alexnet.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model_alexnet.add(BatchNormalization())

model_alexnet.add(Flatten())
model_alexnet.add(Dense(4096,activation='relu'))
model_alexnet.add(Dropout(0.5))
model_alexnet.add(Dense(4096,activation='relu'))
model_alexnet.add(Dropout(0.5))
model_alexnet.add(Dense(5,activation='softmax'))
model_alexnet.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
model_alexnet.summary()




# training
History = model_alexnet.fit(
  train_ds,
  validation_data=val_ds,
  epochs=300,
  callbacks=[EStop]
)


#save model
import time
timestr = time.strftime("%Y%m%d_%H%M%S")
model_alexnet.save('alexnet_model_{}.h5'.format(timestr)) 


# Training History and draw the picture of accuracy and loss
'''
  import collections
  import pandas as pd
  hist = History.history
  
  for key, val in hist.items(): # Count the number of epoch
      numepo = len(np.asarray(val))
      break
  hist = collections.OrderedDict(hist)
  pd.DataFrame(hist).to_excel('model_{}_history.xlsx'.format(timestr), index=True)
  
  import matplotlib.pyplot as plt
  
  plt.plot(History.history['accuracy'])
  plt.plot(History.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'dev'], loc='upper left')
  plt.savefig('Model_alex accuracy_{}.png'.format(timestr))
  plt.show()
  plt.cla()
  
  plt.plot(History.history['loss'])
  plt.plot(History.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'dev'], loc='upper left')
  plt.savefig('Model_alex loss_{}.png'.format(timestr))
  plt.show()
  plt.close()
'''

# for confuse matrix
'''
  import itertools
  from sklearn.metrics import confusion_matrix
  from keras.utils import plot_model
  
  def plot_confusion_matrix(cm, classes_x,classes_y,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
      """
      This function prints and plots the confusion matrix.
      Normalization can be applied by setting `normalize=True`.
      """
      if normalize:
          cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
          print("Normalized confusion matrix")
      else:
          print('Confusion matrix, without normalization')
  
      print(cm)
      plt.clf()
      plt.cla()
      plt.style.use('default')
      plt.imshow(cm, interpolation='nearest', cmap=cmap)
      plt.grid(False)
      plt.title(title)
      plt.colorbar()
      tick_marks_x = np.arange(len(classes_x))
      tick_marks_y = np.arange(len(classes_y))
      plt.xticks(tick_marks_x, classes_x, rotation=45)
      plt.yticks(tick_marks_y, classes_y)
  
      fmt = '.2f' if normalize else 'd'
      thresh = cm.max() / 2.
      for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
          plt.text(j, i, format(cm[i, j], fmt),
                   horizontalalignment="center",
                   color="white" if cm[i, j] > thresh else "black")
  
      plt.tight_layout()
      plt.ylabel('Actual')
      plt.xlabel('Predicted')
      plt.savefig(title+'.png',dpi=350 ,bbox_inches='tight')
      plt.show()
      plt.close()
  
  test_pred  = model.predict(x_test)
  cnf_matrix = confusion_matrix(np.argmax(y_test, axis=1).reshape(-1,1),
                                np.argmax(test_pred, axis=1).reshape(-1,1))
  np.set_printoptions(precision=2)
  plot_confusion_matrix(cnf_matrix,['A','B','C'],['A','B','C'],normalize=True,title='LeNet Confusion Matrix')

'''


