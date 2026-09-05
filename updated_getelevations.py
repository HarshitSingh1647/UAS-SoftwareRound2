import cv2
import numpy as np

inp = "photos/4.jpg"



elev_colour = np.array(
    [
        [0, 215, 84],
        [55, 194, 97],
        [0, 137, 76],
        [0, 82, 47],
    ],
    dtype=np.int16,
)


def classify_shape(contour):

    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)

    x, y, w, h = cv2.boundingRect(contour)


    aspect_ratio = max(w, h) / min(w, h)
    vertices = cv2.approxPolyDP(contour, 0.03 * peri, True)

    if len(vertices) == 3:
        return "triangle"

    if len(vertices) == 4 and aspect_ratio <= 1.25:
        return "square"

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area else 1.0
    if 5 <= len(vertices) <= 12 and solidity < 0.9:
        return "star"

    circularity = 4 * np.pi * area / (peri * peri)
    if circularity >= 0.7 and aspect_ratio <= 1.8:
        return "circle"

    return None






def detect_elevation(image, green_mask,x,y,w,h):
  
 


    ring = np.zeros(green_mask.shape, dtype=np.uint8)
    ring[y - 8:y + h + 8, x - 8:x + w + 8] = green_mask[y - 8:y + h + 8, x - 8:x + w + 8]
    ring[y : y + h, x : x + w] = 0
    pixels = image[ring > 0].astype(np.int16)

    if len(pixels) == 0:
        return None

    distances = ((pixels[:, None, :] - elev_colour[None, :, :]) ** 2).sum(axis=2)
    closest_colors = np.argmin(distances, axis=1)
    counts = np.bincount(closest_colors, minlength=len(elev_colour))
    return int(np.argmax(counts)) + 1



image = cv2.imread(inp)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
 

contours, _ = cv2.findContours(cv2.bitwise_not(green_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

detections = []
for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100 or area > 20000:
            continue

        shape = classify_shape(contour)
        if shape == None or shape == "triangle":
            continue

        x,y,w,h= cv2.boundingRect(contour)
         
        center = (x + w // 2, y + h // 2)
        elevation = detect_elevation(image, green_mask, x,y,w,h)
        detections.append((center, shape, elevation,x,y,w,h ))

detections.sort(key=lambda item: (item[0][1], item[0][0]))
for i, (center, shape, elevation, x,y,w,h) in enumerate(detections, 1):
        
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
     
        cv2.putText(image,f"{shape}, elevation {elevation}",(x, y - 8),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0, 0, 0),2,)
        print(f"item {i}: {shape}, elevation {elevation}, "f"location {center}")

cv2.imwrite("photos/output/elevation_detection.png", image)
print(f"Detected {len(detections)} items")


