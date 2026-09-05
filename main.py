import cv2
import numpy as np


inp = "photos/4.jpg"


elev_colours = np.array([ [0, 215, 84],[55, 194, 97],[0, 137, 76],[0, 82, 47],])


def Fshape(contour):
    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    

    _, _, w, h = cv2.boundingRect(contour)
    
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
    if circularity >= 0.7 and aspect_ratio <= 1.2:
        return "circle"

    return None


def detect_elevation(image, g_mask, x, y, w, h):
    ring = np.zeros(g_mask.shape, dtype=np.uint8)
    ring[
        max(0, y - 8) :  y + h + 8,
        max(0, x - 8) : x + w + 8,
    ] = g_mask[
        max(0, y - 8) : y + h + 8,
        max(0, x - 8) : x + w + 8,
    ]
    ring[y : y + h, x : x + w] = 0
    pixels = image[ring > 0].astype(np.int16)
    if len(pixels) == 0:
        return None

    distances = ((pixels[:, None, :] - elev_colours[None, :, :]) ** 2).sum(axis=2)
    closest_colors = np.argmin(distances, axis=1)
    counts = np.bincount(closest_colors, minlength=len(elev_colours))
    return int(np.argmax(counts)) + 1


def detect_casualities(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    g_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    object_mask = cv2.bitwise_not(g_mask)
    contours, _ = cv2.findContours(
        object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    casualities = []
    start_end = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100 or area > 20000:
            continue

        shape = Fshape(contour)
        if shape is None:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)

        if shape == "triangle":
            b, g, r = image[center[1], center[0]]
            if 30 <= g <= 42 and 110 <= b <= 120 and r >= 240:
                start_end.insert(0, center)
            else:
                start_end.append(center)
            continue

        b, g, r = image[center[1], center[0]]
        if r >= 200 and g <= 50 and b <= 50:
            severity = 3
        elif r >= 200 and g >= 200 and b >= 200:
            severity = 1
        elif 50 <= r <= 255 and 60 <= g <= 100 and b >= 200:
            severity = 2
        else:
            severity = 0

        age = {"star": 1, "circle": 3, "square": 2}[shape]
        elevation = detect_elevation(image, g_mask, x, y, w, h)
        casualities.append(
            {
                "shape": shape,
                "severity_score": severity,
                "age_score": age,
                "priority_score": severity * age,
                "elevation": elevation,
                "location": center,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        )

    return casualities, start_end



image = cv2.imread(inp)
if image is None:
    raise FileNotFoundError(f"Could not read {inp}")

casualities, start_end = detect_casualities(image)
casualities.sort(key=lambda item: item["priority_score"], reverse=True)

for number, casuality in enumerate(casualities, 1):
    x = casuality["x"]
    y = casuality["y"]
    w = casuality["w"]
    h = casuality["h"]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
    label = f"{casuality['shape']}, elevation {casuality['elevation'] or '?'}"
    cv2.putText(image,label,(x, max(20, y - 8)),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0, 0, 0),2,)
    print(f"casuality{number}: {casuality['shape']}, "f"severity {casuality['severity_score']}, "f"age {casuality['age_score']}, "f"priority {casuality['priority_score']}, "f"elevation {casuality['elevation'] or 'unknown'}, "f"location {casuality['location']}")

if casualities:
    if start_end:
        cv2.line(image, start_end[0], casualities[0]["location"], (0, 0, 0), 3)

    for previous, current in zip(casualities, casualities[1:]):
        cv2.line(image, previous["location"], current["location"], (0, 0, 0), 3)

    if len(start_end) >= 2:
        cv2.line(image, casualities[-1]["location"], start_end[1], (0, 0, 0), 3)

cv2.imwrite("photos/output/main.png", image)
print(f"Detected {len(casualities)} casualities")


