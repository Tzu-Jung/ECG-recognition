from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys
import numpy as np
from tensorflow.keras import layers

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
import os
import csv
from PIL import Image
from cv2 import cv2
img_height = 100
img_width = 100
size = (img_width,img_height)

# get path of picture from parameter
files = sys.argv[1]

# load the model
net_first = load_model('densenet_model.h5')
#net_normalize = load_model('batchnormalize.h5')
net_early_stop = load_model('vgg_model.h5')
#net_300_epochs = load_model('vgg300.h5')
#net_vgg = load_model('model_vgg.h5')


pred_inds = []

with open('output_.csv', 'w', newline='') as csvfile:
# build the CSV writer
    writer = csv.writer(csvfile)

    # write the title   
    writer.writerow(['ImageID', 'PredictedLabel'])
    i = 0
    counter = [0, 0, 0, 0, 0] # QB, ann, egg, iron, kiba
    QB = 0
    ann = 1
    egg = 2
    iron = 3
    kiba = 4
    # identify the picture
    for f in os.listdir(files):

        img = Image.open(os.path.join(files, f))
        
        img = img.resize(size,Image.BILINEAR)
        imgarray = np.array(img)
        imgarray = imgarray[:, :, :3]

        imgarray = np.expand_dims(imgarray, axis = 0)
        
        
        
        #pred = net_first.predict_classes(imgarray)[0]
        first = net_first.predict(imgarray)[0]
        predict_classes=np.argmax(first,axis=0)
        
        #pred = net_normalize.predict_classes(imgarray)[0]
        #normalize = net_normalize.predict(imgarray)[0]

        #pred[2] = net_densenet.predict_classes(imgarray)[0]
        early_stop = net_early_stop.predict(imgarray)[0]
        early = net_early_stop.predict_classes(imgarray)[0]

        #pred = net_resnet.predict_classes(imgarray)[0]
        #epochs_300 = net_300_epochs.predict(imgarray)[0]

        #pred = net_vgg.predict_classes(imgarray)[0]
        #vgg = net_vgg.predict(imgarray)[0]
        #print(f)
        
        ans = [early_stop, early_stop, first]
        final = np.sum(ans,axis = 0).argmax()

        
        if final == 0 :
            pred_ = 'QB'
            counter[QB] = counter[QB] + 1
        elif final == 1:
            pred_ = 'ann'
            counter[ann] = counter[ann] + 1
        elif final == 2:
            pred_ = 'egg'
            counter[egg] = counter[egg] + 1
        elif final == 3:
            pred_ = 'iron'
            counter[iron] = counter[iron] + 1
        else :
            pred_ = 'kiba'
            counter[kiba] = counter[kiba] + 1
        '''
          if predict_classes == 0 :
              pred_ = 'QB'
          elif predict_classes == 1:
              pred_ = 'ann'
          elif predict_classes == 2:
              pred_ = 'egg'
          elif predict_classes == 3:
              pred_ = 'iron'
          else :
              pred_ = 'kiba'
        '''
        writer.writerow([f, pred_])
        i = i + 1
        
        if i % 10 == 0:
          sm = np.argmax(counter)
          if sm == 0 :
              total_ans = 'QB'
          elif sm == 1:
              total_ans = 'ann'
          elif sm == 2:
              total_ans = 'egg'
          elif sm == 3:
              total_ans = 'iron'
          else :
              total_ans = 'kiba'
          writer.writerow(['Max', total_ans, np.amax(counter)])
          counter = [0, 0, 0, 0, 0]
          i = 0
        else :
            continue
    
print("f end")

       
