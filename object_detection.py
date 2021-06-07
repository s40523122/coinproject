import tensorflow.keras
#from PIL import Image, ImageOps
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
num = 0

while(True):


	# Replace this with the path to your image
	#image = Image.open('test_photo.jpg')
	ret, image = cap.read()

	#resize the image to a 224x224 with the same strategy as in TM2:
	#resizing the image to be at least 224x224 and then cropping from the center
	size = (224, 224)
	#image = ImageOps.fit(image, size, Image.ANTIALIAS)
	image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

	#turn the image into a numpy array
	image_array = np.asarray(image)

	# display the resized image
	#image.show()

	# Normalize the image
	normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

	# Load the image into the array
	data[0] = normalized_image_array

	# run the inference
	prediction = model.predict(data)
	#print(prediction)

	box = {0 : "Stop", 1 : "Start", 2 : "Right", 3 : "Left"}
	index = np.argmax(prediction)

	if num == 10:
		print("\n辨識結果 :\n", box[index])
		print(round(np.max(prediction)*100, 1), "%")
		num =0
	else:
		num+= 1

	cv2.imshow('frame', image)

	# 若按下 q 鍵則離開迴圈
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
