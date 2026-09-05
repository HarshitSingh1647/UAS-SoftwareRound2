import cv2
import numpy as np
#hsv(260.47, 91.06%, 92.16%)
#hsv(258.44, 87.23%, 92.16%)
#, 30, 
image = cv2.imread("./photos/4.jpg")
blank=cv2.bitwise_not(np.zeros_like(image))
l_black = np.array([0, 0, 0])
u_black = np.array([40, 40, 40])
obstacle_mask = cv2.inRange(image, l_black, u_black)
water_mask = cv2.inRange(image, np.array([200, 10,70 ]), np.array([250, 50, 120]))
cv2.imshow("obstacle_mask", cv2.bitwise_not(obstacle_mask))
cv2.waitKey(0)
cv2.imshow("water_mask", cv2.bitwise_not(water_mask))
cv2.waitKey(0)

non_mask = cv2.bitwise_or(obstacle_mask,water_mask)
cv2.imshow("non_mask", cv2.bitwise_not(non_mask))
cv2.waitKey(0)

cv2.imwrite("obstacle_mask.png", obstacle_mask)
