import cv2
# i dont know why it dosen't detect one square in image 3 , rest is fine 
inp="photos/3.jpg"

def Fshape(contour):
    peri = cv2.arcLength(contour, True)

    if peri == 0:
        return None

    area = cv2.contourArea(contour)
    verti = cv2.approxPolyDP(contour, 0.04 * peri, True)
 
    _, _, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / min(w, h)

    if len(verti) == 3:
        return "triangle"

    if len(verti) == 4:
        if aspect_ratio <= 1.2:
            print("square")
            return "square"
        print("square2")
        return None
    
    print("square4") #wrote this to see what it does when above 2 are not detected
  
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area else 1
    if (6 <= len(verti) <= 11 and solidity < 0.9 and max(w, h) <= 120):
        return "star"

    
    if (4 * 3.14 * area )/ (peri * peri) >= 0.7 and aspect_ratio <= 1.2:
        return "circle"
    return None



image = cv2.imread(inp)
    
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
gbg= cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
non_green = cv2.bitwise_not(gbg)
  
    
edges = cv2.Canny(image, 50, 150)
   
mask = cv2.bitwise_or(non_green, edges)


contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  

for i in contours:
    if cv2.contourArea(i) < 100:
        continue

    x, y, w, h = cv2.boundingRect(i)



    #print(type(cv2.boundingRect(i)))



    shape = Fshape(i)
    if shape is None:
        continue


    cv2.rectangle(image,(x, y),(x+w, y+h),(0, 0, 0),2,)
  
    cv2.putText(image,shape,(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0),2,)
   

cv2.imwrite("photos/output/shape_detect.png", image)


   
 



