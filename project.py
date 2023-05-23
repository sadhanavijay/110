# To Capture Frame
import cv2

# To process image array
import numpy as np

# import the tensorflow modules and load the model
import tensorflow as tf

model=tf.keras.models.load_model('keras_model.h5')

# Attaching Cam indexed as 0, with the application software
vid = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the vid 
	status , frame = vid.read()

	# if we were sucessfully able to read the frame
	if status:
	
		frame = cv2.flip(frame , 1)
	
		resize_img = cv2.resize(frame, (224,224))

		# img = np.array(resize_img, dtype = np.float32)
		
		dimension = np.expand_dims(resize_img, axis=0)
		normal = dimension/255
		prediction = model.predict(normal)
		scissor = int(prediction[0][0]*100)
		rock = int(prediction[0][1]*100)
		paper = int(prediction[0][2]*100)
		print('prediction: \nscissor{}\nrock{}\nPaper{}'.format(scissor,rock,paper))
		cv2.imshow('feed' , frame)
		key = cv2.waitKey(1)
	
		if key == 32:
			break

# release the vid from the application software
vid.release()

# close the open window
cv2.destroyAllWindows()
