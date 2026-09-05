import cv2
import numpy as np
# i dont know why it dosen't detect one square in image 3 , rest is fine 
inp="photos/6.jpg"

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
            #print("square")
            return "square"
        #print("square2")
        return None
   
    
    #print("square4") #wrote this to see what it does when above 2 are not detected
  
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area else 1
    if (6 <= len(verti) <= 11 and solidity < 0.9 and max(w, h) <= 120):
        return "star"

    
    if (4 * 3.14 * area )/ (peri * peri) >= 0.7 and aspect_ratio <= 1.2:
        return "circle"
    return None



image = cv2.imread(inp)
total_casuality=0
#print(image[20,20])
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
gbg= cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
non_green = cv2.bitwise_not(gbg)
  
    
edges = cv2.Canny(image, 50, 150)
   
mask = cv2.bitwise_or(non_green, edges)


contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
casualities = []
start_end=[]
i_count=0
for countour in contours:
    
    if cv2.contourArea(countour) < 100 or cv2.contourArea(countour) > 20000:
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
    else:
        severity = 0

    shape = Fshape(countour)


    if shape is None:
        continue

    i_count+=1
    if shape == "star":
        age=1
    if shape == "circle":
        age=3
    if shape == "square":
        age=2
    if shape == "triangle":
        #252, 116, 36
        if 30<=g <= 42 and 110<=b <= 120 and 240<=r <= 255:
            cv2.putText(image,"Start",(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0),2,)
            start_end.insert(0,(int(x+w/2), int(y+h/2)))
        else:
            cv2.putText(image,"End",(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0),2,)
            start_end.insert(1,(int(x+w/2), int(y+h/2)))
        continue


    total_casuality+=1
    
    casualities.append({
        "casuality_no": i_count,
        "severity_score": severity,
        "age_score": age,
        "priority_score": severity * age,
        "location": (int(x + w / 2), int(y + h / 2))
    })


    cv2.rectangle(image,(x, y),(x+w, y+h),(0, 0, 0),2,)
  
    cv2.putText(image,shape,(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0),2,)

    print(f"casuality no.{i_count}, severity score - {severity}, age score - {age} , priority score - {severity*age}, location - ({int(x+w/2)} , {int(y+h/2)})")

print("total no. of casuality",total_casuality)


casualities.sort(key=lambda x: x["priority_score"], reverse=True)


if casualities:
    cv2.line(image, start_end[0], casualities[0]["location"], (0, 0, 0), thickness=3, lineType=cv2.LINE_8)

    for casuality in range(1, len(casualities)):
        
        
        cv2.line(
            image,
            casualities[casuality - 1]["location"],
            casualities[casuality]["location"],
            (0, 0, 0),
            thickness=3,
            lineType=cv2.LINE_8,
        )

    cv2.line(image, casualities[-1]["location"], start_end[1], (0, 0, 0), thickness=3, lineType=cv2.LINE_8)




cv2.imwrite("photos/output/connected.png", image)


   
 



