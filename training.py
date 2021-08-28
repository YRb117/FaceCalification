#Programa para el entrenamiento de la red neuronal a partir de las imagenes guardadas y sus etiquetas. 
import cv2
import os
import numpy as np 
import time
from random import shuffle
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

dataPath = 'data4'
peoplelist = os.listdir(dataPath)
print('Lista de personas: ', peoplelist)

img_height = 50
img_width =50

def my_data():
	labels = []
	facesData = []
	label = 0

	for nameDir in peoplelist: 
		personPath = dataPath + '/' + nameDir
		print('Leyendo las imágenes')
		for fileName in os.listdir(personPath):
			#print('Rostros: ', nameDir + '/' + fileName)
			labels.append(label)
			image = cv2.imread(personPath+'/'+fileName,cv2.IMREAD_GRAYSCALE)
			image = cv2.resize(image,(50,50))
			facesData.append([np.array(image),label])
			#print(np.array(facesData).shape)
			#cv2.imshow('image',image)
			#cv2.waitKey(10)
		label = label + 1
	shuffle(facesData)
	print(np.array(facesData, dtype="object").shape)
	return facesData

data = my_data()


train = data[:1200]  
test = data[1200:]
X_train = np.array([i[0] for i in train],dtype="object").reshape(-1,50,50,1)
print(X_train.shape)
y_train = [i[1] for i in train]
y_train = np.array(y_train)
X_test = np.array([i[0] for i in test],dtype="object").reshape(-1,50,50,1)
print(y_train.shape)
y_test = [i[1] for i in test]
y_test = np.array(y_test)

#X_train = data_augmentation(X_train)
#X_test = data_augmentation(X_test)

X_train, X_test = X_train/255.0, X_test/255.0


model = models.Sequential([
	layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(50, 50,1)),
	layers.MaxPooling2D((2, 2)),
	layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
	layers.MaxPooling2D((2, 2)),
	layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
	layers.MaxPooling2D((2, 2)),
	layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
	layers.MaxPooling2D((2, 2)),
	layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
	layers.MaxPooling2D((2, 2)),
	layers.Flatten(),
	layers.Dense(1024, activation='relu'),
	layers.Dropout(0.7),
	layers.Dense(5, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

model.save('modelFinal2.h5')