#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist


# In[2]:


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[4]:


#Arquitectura del modelo


# In[5]:


from keras import models
from keras import layers


# In[6]:


model = models.Sequential()
model.add(layers.Dense(512, activation = 'relu', input_shape = (28*28,)))
model.add(layers.Dense(10, activation = 'softmax'))


# In[7]:


#Etapa de compilación


# In[8]:


model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[9]:


#Etapa de preparación de las imágenes


# In[10]:


train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255


# In[11]:


test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255


# In[12]:


#Etapa de preparación de las etiquetas


# In[13]:


from keras.utils import to_categorical


# In[14]:


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[17]:


#Etapa de entrenamiento


# In[ ]:


model.fit(train_images, train_labels, epochs=5, batch_size = 128)


# In[ ]:




