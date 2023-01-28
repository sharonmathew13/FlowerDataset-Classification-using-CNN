#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:


import zipfile as zf
files = zf.ZipFile("flowers (2).zip", 'r')
files.extractall('flowers')
files.close()


# In[4]:


import os
import cv2
import tqdm as tqdm
print(os.listdir('flowers/flowers'))


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import tensorflow as tf
import random as rn
import cv2 
import numpy as np 
from tqdm import tqdm


# In[9]:


X=[]
Z=[]
IMG_SIZE=150
FLOWER_DAISY_DIR='flowers/flowers/daisy'
FLOWER_SUNFLOWER_DIR='flowers/flowers/sunflower'
FLOWER_TULIP_DIR='flowers/flowers/tulip'
FLOWER_DANDI_DIR='flowers/flowers/dandelion'
FLOWER_ROSE_DIR='flowers/flowers/rose'


# In[12]:


def assign_label(img,flower_type):
    return flower_type


# In[13]:


def make_train_data(flower_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,flower_type)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        X.append(np.array(img))
        Z.append(str(label))


# In[14]:


make_train_data('Daisy',FLOWER_DAISY_DIR)
print(len(X))
make_train_data('Sunflower',FLOWER_SUNFLOWER_DIR)
print(len(X))
make_train_data('Rose',FLOWER_ROSE_DIR)
print(len(X))
make_train_data('Dandelion',FLOWER_DANDI_DIR)
print(len(X))


# In[15]:


make_train_data('Tulip',FLOWER_TULIP_DIR)
print(len(X))


# In[18]:



fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(Z))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('Flower: '+Z[l])
 
plt.tight_layout()


# In[22]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,5)
X=np.array(X)
X=X/255
x_train_val,X_test,y_train_val,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)
np.random.seed(42)
rn.seed(42)


# In[30]:


x_train_val


# In[23]:


from keras.models import Sequential
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense


# In[24]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (150,150,3), padding='same', activation = 
'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation = 'softmax'))


# In[27]:


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
 featurewise_center=False, # set input mean to 0 over the dataset
 samplewise_center=False, # set each sample mean to 0
 featurewise_std_normalization=False, # divide inputs by std of the dataset
 samplewise_std_normalization=False, # divide each input by its std
 zca_whitening=False, # apply ZCA whitening
 rotation_range=10, # randomly rotate images in the range (degrees, 0 to 180)
 zoom_range = 0.1, # Randomly zoom image 
 width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
 height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
 horizontal_flip=True, # randomly flip images
 vertical_flip=False) # randomly flip images


# In[35]:


datagen.fit(x_train_val)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
History = model.fit_generator(datagen.flow(x_train_val,(y_train_val), batch_size=32),
 epochs = 5, validation_data = (X_test,y_test),
verbose = 1, steps_per_epoch=x_train_val.shape[0] // 32)


# In[ ]:




