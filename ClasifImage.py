import cv2
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential


dataPath = 'data4' 
imagePaths = os.listdir(dataPath)
print('imagePaths = ', imagePaths)

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
new_model = keras.models.load_model('modelFinal2.h5')

frame = cv2.imread('Panorama5.jpg')
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
auxFrame = gray.copy()

faces = faceClassif.detectMultiScale(gray,
	scaleFactor=1.15,
	minNeighbors=5,)

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



cv2.imshow('image',frame)

cv2.waitKey(0)
cv2.destroyAllWindows()