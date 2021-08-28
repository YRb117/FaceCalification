# FaceCalification
Using OpenCV, faces are detected in images or video and then classified using convolutional neural network


To use the face classifier it is necessary.

-Run the captureRostro.py file to detect and save the user's faces. Run as many people as you want to capture by changing the user label.

-Run training.py to train the neural network from the previously saved faces. Make sure to update the location of the images. The trained model will be saved.

-Run califfacesVideo.py or ClasifImage.py to detect and classify faces in image or video. Make sure to update the name of the trained model and the location of any known tags.
