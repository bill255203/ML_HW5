#!/usr/bin/env python
# coding: utf-8

# # 2022 Machine Learning CA5 Demo

# ## I. LeNet on MNIST Dataset

# ![image.png](attachment:71329533-85ed-4a9d-a314-8312be2d3895.png)

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Flatten, Dense, Softmax
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model


# In[2]:


tf.__version__


# ### 1. Load Dataset: MNIST

# In[3]:


(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train=x_train/255
x_test=x_test/255


# In[4]:


x_train.shape
# x_train = np.expand_dims(x_train , -1)
x_train = x_train.reshape(*x_train.shape,1)
x_train.shape


# #### Visualization

# In[5]:


plt.figure(figsize = (5,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(x_train[i], cmap = 'gray')
    plt.xticks([])
    plt.yticks([])
plt.show()


# ### 2. Model: LeNet

# #### 2.1 Build Model

# Method 1. Functional

# In[6]:


# *Note: You don't have to build it with a class like how I did.
class LeNet(Model):
    def __init__(self, input_shape = (28,28,1), output_shape = 10):
        kernel_size = 5
        input_layer = Input(shape = input_shape)
        x = Conv2D(6, kernel_size, padding = 'same', activation='relu')(input_layer)
        x = MaxPooling2D((2, 2))(x)
        
        x = Conv2D(16, kernel_size, activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)
        x = Dense(120, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        output_layer = Dense(output_shape, activation = 'softmax')(x)
        
        super().__init__(input_layer, output_layer, name = 'LeNet')
        self.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])


# In[7]:


model = LeNet()
model.summary()


# Method 2. Sequential

# In[8]:


from tensorflow.keras.models import Sequential
input_shape = (28,28,1)
output_shape = 10
model = Sequential([
    Input(shape = input_shape),
    Conv2D(6, 5, padding = 'same', activation='relu'),
    MaxPooling2D((2, 2)), 
    Conv2D(16, 5, activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(64, activation='relu'),
    Dense(output_shape, activation = 'softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# #### 2.2 Tain Model

# In[9]:


history = model.fit(x_train,y_train,
                    batch_size = 128,
                    epochs = 20,
                    validation_split = 0.2,
                    # validation_data = (x_val,y_val)
                    )


# In[10]:


# show train history
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()


# In[11]:


# show train history
plt.plot(history.history['accuracy'], label = 'acc')
plt.plot(history.history['val_accuracy'], label = 'val_acc')
plt.legend()
plt.show()


# The validation loss stoped falling early at about the 5~10th epoch.  
# Further training may cause overfitting!!!

# #### 2.3 EarlyStopping

# In[12]:


from tensorflow.keras.callbacks import EarlyStopping
model = LeNet()
early_stop = EarlyStopping(monitor = 'val_loss', patience=3, min_delta = 0.01)
history = model.fit(x_train,y_train,
                    batch_size = 128,
                    epochs = 20,
                    validation_split = 0.2,
                    # validation_data = (x_val,y_val)
                    callbacks = [early_stop]
                    )


# In[ ]:


# save model
model.save('model.h5')


# In[13]:


# show train history
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()


# In[14]:


# show train history
plt.plot(history.history['accuracy'], label = 'acc')
plt.plot(history.history['val_accuracy'], label = 'val_acc')
plt.legend()
plt.show()


# The model stoped training after finding out the validation loss didn't seem to decrease anymore.

# ### 3. Results

# #### 3.1 Accuracy

# In[15]:


from sklearn.metrics import accuracy_score

pred_train = model.predict(x_train).argmax(1)
pred_test = model.predict(x_test).argmax(1)

print('Train Accuracy:', accuracy_score(pred_train, y_train))
print('Test Accuracy:', accuracy_score(pred_test, y_test))


# #### 3.2 Confusion Matrix

# In[16]:


import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(true ,pred, classes, title = 'confusion_matrix'):
    plt.figure(figsize = (5,5))
    sns.heatmap(confusion_matrix(true, pred),
                square= True, annot=True, cbar= False, fmt = '.20g')
    plt.title(title)
    plt.xlabel("predicted value")
    plt.ylabel("true value")
    plt.xticklabels = classes
    plt.yticklabels = classes
    plt.show()    


# In[17]:


plot_confusion_matrix(y_test,pred_test, classes=range(10),
                      title='Confusion matrix')


# In[ ]:





# ## II. Data Augmentation

# ### 1. Flip

# In[18]:


# Example-Flip
def augmentation(images, lables): # flip
    images_ = np.concatenate([images, np.flip(images, axis=-1)])
    lables_ = np.concatenate([lables,lables])
    return images_, lables_


# results will be like......

# In[19]:


img = plt.imread('demo.png')
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(np.flip(img, axis = 1))
plt.show()


# ### 2. Zoom In/Zoom Out

# In[20]:


# Example-Zoom
import cv2
def zoom(img, zoom_factor = 1):
    size_ = tuple(int(x*zoom_factor) for x in img.shape[:2])
    zoomed = cv2.resize(img, size_, 
                        interpolation=cv2.INTER_CUBIC)
    _ = abs(zoomed.shape[0]-img.shape[0])
    edge = [_//2, _-_//2]
    if zoom_factor>1:
        zoomed = zoomed[edge[0]:zoomed.shape[0]-edge[1],
                        edge[0]:zoomed.shape[0]-edge[1]]
        return zoomed.clip(img.min(),img.max())
    elif zoom_factor<1:
        zoomed = np.stack([np.pad(img[:,:,c], edge, mode = 'edge') for c in range(3)],
                          axis = -1)
        return  zoomed.clip(img.min(),img.max())
    else: 
        return img


# In[21]:


zoomed = []
plt.figure(figsize = (7,2))
for i,v in enumerate([0.8,1.0,1.2],1):
    plt.subplot(1,3,i)
    plt.xlabel(f'x {v}')
    plt.imshow(zoom(img, v))


# In[22]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
     rotation_range=20,
     width_shift_range=0.2,
     height_shift_range=0.2,
    zoom_range=(0.8,1.4),
     horizontal_flip=True
    )
datagen.fit(np.expand_dims(img,0))
a = datagen.flow(np.expand_dims(img,0),np.array([1])).next()
plt.imshow(a[0])
plt.show()

# In[23]:


aug_gen = datagen.flow(np.expand_dims(img,0),np.array([1]))
plt.figure(figsize = (12,6))
for i in range(15):
    img_,_ = aug_gen.next()
    
    plt.subplot(3,5,i+1)
    plt.imshow(img_[0])
plt.show()


# In[ ]:




