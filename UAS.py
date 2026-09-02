import cv2
import numpy as np


image = cv2.imread("./photos/1.jpg")

l_black = np.array([0, 0, 0])
u_black = np.array([40, 40, 40])
obstacle_mask = cv2.inRange(image, l_black, u_black)
print(obstacle_mask)

obstacle_mask = cv2.bitwise_not(obstacle_mask)
cv2.imshow("obstacle_mask", obstacle_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("obstacle_mask.png", obstacle_mask)

