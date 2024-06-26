from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Arquitectura del modelo

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(512, activation = 'relu', input_shape = (28*28,)))
model.add(layers.Dense(10, activation = 'softmax'))

#Etapa de compilación

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Etapa de preparación de las imágenes

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

#Etapa de preparación de las etiquetas

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Etapa de entrenamiento

model.fit(train_images, train_labels, epochs=5, batch_size = 128)
