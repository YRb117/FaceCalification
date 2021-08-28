import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import cv2
import os 
import numpy as np

new_model = keras.models.load_model('model7.h5')

dataPath = 'data2' 
imagePaths = os.listdir(dataPath)
print('imagePaths = ', imagePaths)

cap = cv2.VideoCapture(0)

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
	ret,frame = cap.read()
	if ret == False: break
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	auxFrame = gray.copy()
	faces = faceClassif.detectMultiScale(gray,1.1,8)
	for(x,y,w,h) in faces:
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(50,50),interpolation=cv2.INTER_CUBIC)
		rostro = np.expand_dims(np.array(rostro/255), axis=0)
		rostro = np.expand_dims(rostro, axis=3)
		#print(rostro.shape)
		result = new_model.predict(rostro)
		res = np.max(np.array(result))
		label = np.argmax(np.array(result))
		#print(res)
		cv2.putText(frame,'{}'.format(res),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
		if res > 0.8:
			cv2.putText(frame,'{}'.format(imagePaths[label]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		else:
			cv2.putText(frame,'Desconocido',(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
	cv2.imshow('frame',frame)
	k = cv2.waitKey(1)
	if k == 27:
		break
print('Fin')

cap.release()
cv2.destroyAllWindows()