import numpy as np
import cv2

img_target = "Donnees/I4.jpg" 
img_source = "Donnees/I2.jpg"

# loading images
targetbgr = cv2.imread(img_target, cv2.IMREAD_UNCHANGED)
sourcebgr = cv2.imread(img_source, cv2.IMREAD_UNCHANGED)

# converting images to LAB space
target = cv2.cvtColor(targetbgr,cv2.COLOR_BGR2LAB)
source = cv2.cvtColor(sourcebgr,cv2.COLOR_BGR2LAB)

# computing standart deviation and mean on the 3 channels separatly
mean_target, std_target = cv2.meanStdDev(target)
mean_target =  np.hstack(np.around(mean_target,2)) 
std_target = np.hstack(np.around(std_target,2))

mean_source, std_source = cv2.meanStdDev(source)
mean_source = np.hstack(np.around(mean_source,2))
std_source = np.hstack(np.around(std_source,2))

# transfering colors
result = ((source-mean_source)*(std_target/std_source))+mean_target

# dealing with problematic pixels
height, width, channel = source.shape
for i in range(0,height):
	for j in range(0,width):
		for k in range(0,channel):
			x = result[i, j, k]
			x = round(x)
			# boundary check
			x = 0 if x<0 else x
			x = 255 if x>255 else x
			result[i, j, k] = x

# convert back to RGB and save result
converted = cv2.cvtColor(result.astype('uint8'),cv2.COLOR_LAB2BGR)
cv2.imwrite('Donnees/result.jpg',converted)
