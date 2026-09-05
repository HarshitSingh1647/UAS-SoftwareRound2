import cv2
import numpy as np


#1=84, 212, 1
#2=97, 194, 58
#3=77, 135, 0
#4=51, 82, 1


image= cv2.imread("photos/4.jpg")

elevation_1=cv2.inRange(image, np.array([0, 195,75 ]), np.array([10, 240,95 ]))
elevation_1_path=cv2.bitwise_and(image,image,mask=elevation_1)
cv2.imshow("elevation_1_path", elevation_1_path)
cv2.waitKey(0)


elevation_2=cv2.inRange(image, np.array([50, 186,93 ]), np.array([65, 204,110 ]))
elevation_2_path=cv2.bitwise_and(image,image,mask=elevation_2)
cv2.imshow("elevation_2_path", elevation_2_path)
cv2.waitKey(0)


elevation_3=cv2.inRange(image, np.array([0, 125,70 ]), np.array([5, 143,85 ]))
elevation_3_path=cv2.bitwise_and(image,image,mask=elevation_3)
cv2.imshow("elevation_3_path", elevation_3_path)
cv2.waitKey(0)


elevation_4=cv2.inRange(image, np.array([0, 75,45 ]), np.array([5, 90,60 ]))
elevation_4_path=cv2.bitwise_and(image,image,mask=elevation_4)
cv2.imshow("elevation_4_path", elevation_4_path)
cv2.waitKey(0)