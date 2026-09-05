import cv2
import numpy as np


inp="photos/4.jpg"

#1=84, 212, 1
#2=97, 194, 58
#3=77, 135, 0
#4=51, 82, 1

def Fshape(contour):
    peri = cv2.arcLength(contour, True)

    if peri == 0:
        return None

    area = cv2.contourArea(contour)
    verti = cv2.approxPolyDP(contour, 0.04 * peri, True)
 
    _, _, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / min(w, h)

    if len(verti) == 4:
        if aspect_ratio <= 1.2:
            #print("square")
            return "square"
        #print("square2")
        return None

  
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area else 1
    if (6 <= len(verti) <= 11 and solidity < 0.9 and max(w, h) <= 120):
        return "star"

    
    if (4 * 3.14 * area )/ (peri * peri) >= 0.7 and aspect_ratio <= 1.2:
        return "circle"
    return None



image = cv2.imread(inp)
total_casuality=0

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
gbg= cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
non_green = cv2.bitwise_not(gbg)
edges = cv2.Canny(image, 50, 150)
mask = cv2.bitwise_or(non_green, edges)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)








elevation_2 = np.zeros_like(image)
elevation_2_mask=cv2.inRange(image, np.array([50, 186,93 ]), np.array([65, 204,110 ]))

elevation2_countours, _ = cv2.findContours(elevation_2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for countour in elevation2_countours:
    cv2.drawContours(elevation_2, [countour], -1, (0, 255, 0), 2)







elevation_3 = np.zeros_like(image)
elevation_3_mask=cv2.inRange(image, np.array([0, 125,70 ]), np.array([5, 143,85 ]))

elevation3_countours, _ = cv2.findContours(elevation_3_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for countour in elevation3_countours:
    cv2.drawContours(elevation_3, [countour], -1, (0, 255, 0), 2)






elevation_4 = np.zeros_like(image)
elevation_4_mask=cv2.inRange(image, np.array([0, 75,45 ]), np.array([5, 90,60 ]))

elevation4_countours, _ = cv2.findContours(elevation_4_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for countour in elevation4_countours:
    cv2.drawContours(elevation_4, [countour], -1, (0, 255, 0), 2)

print("elevation4_countours",len(elevation4_countours))


  
index_countour=0
for countour in contours:
    
    if cv2.contourArea(countour) < 100:
        continue

    x, y, w, h = cv2.boundingRect(countour)
    colour = image[int(y+(h/2)), int(x+(w/2))]
   
    g,b, r = colour


    if g <= 50 and b <= 50 and 200<=r <= 255:
        severity = 3

    elif 200<=g <= 255 and 200<=b <= 255 and 200<=r <= 255:
        severity = 1

    elif 60<=g <= 100 and 200<=b <= 255 and 50<=r <= 255:
        severity = 2






    shape = Fshape(countour)


    if shape is None:
        continue
    
    index_countour+=1
    if shape == "star":
        age=1
    if shape == "circle":
        age=3
    if shape == "square":
        age=2


    detected_elevation = None
    for i1 in range(0,len(elevation2_countours)-1):
      
        if cv2.pointPolygonTest(elevation2_countours[i1], (x + w / 2, y + h / 2), False) >= 0:
            
            detected_elevation = 2
            break
            
    if detected_elevation == None :
        for i2 in range(0,len(elevation3_countours)-1):
            if cv2.pointPolygonTest(elevation3_countours[i2], (x + w / 2, y + h / 2), False) >= 0:
                    
                detected_elevation = 3
                break
    if detected_elevation == None:
        for i3 in range(0,len(elevation4_countours)-1):
            if cv2.pointPolygonTest(elevation4_countours[i3], (x + w / 2, y + h / 2), False) >= 0:
                    
                detected_elevation = 4
                break
    else:
        detected_elevation = 1
    
    
    total_casuality+=1
    print(f"casuality no.{index_countour}, severity score - {severity}, age score - {age} , priority score - {severity*age}, location - ({int(x+w/2)} , {int(y+h/2)}), elevation detected - {detected_elevation}")
    
    




print("total no. of casuality",total_casuality)


   





