import heapq
import os
from typing import List, Tuple

import cv2 as cv
import numpy as np


Point = Tuple[int, int]


def detect_markers(image: np.ndarray):
   
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    green_mask = cv.inRange(hsv, lower_green, upper_green)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv.inRange(hsv, lower_red1, upper_red1) | cv.inRange(hsv, lower_red2, upper_red2)

    start = None
    goal = None

    for mask, color_name in [(green_mask, 'green'), (red_mask, 'red')]:
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest = max(contours, key=cv.contourArea)
        area = cv.contourArea(largest)
        if area < 30:
            continue

        M = cv.moments(largest)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        if color_name == 'green':
            start = (cx, cy)
        else:
            goal = (cx, cy)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    obstacle_mask = cv.inRange(gray, 0, 80)
    marker_mask = cv.bitwise_or(green_mask, red_mask)
    obstacle_mask = cv.bitwise_and(obstacle_mask, cv.bitwise_not(marker_mask))
    obstacle_mask = cv.morphologyEx(obstacle_mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    obstacle_mask = cv.morphologyEx(obstacle_mask, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv.findContours(obstacle_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    obstacles = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area < 30:
            continue
        x, y, w, h = cv.boundingRect(contour)
        obstacles.append((x, y, w, h))

    if start is None or goal is None:
        raise ValueError('Could not detect both a start and destination marker in the image.')

    return start, goal, obstacles


def is_walkable(image: np.ndarray, point: Point, obstacles: List[Tuple[int, int, int, int]]) -> bool:
    x, y = point
    if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
        return False

    for ox, oy, ow, oh in obstacles:
        if ox <= x <= ox + ow and oy <= y <= oy + oh:
            return False

    return True


def find_shortest_path(image: np.ndarray, start: Point, goal: Point, obstacles: List[Tuple[int, int, int, int]]):
   
    h, w = image.shape[:2]
    visited = set()
    pq = []
    came_from = {}
    g_score = {start: 0}
    f_score = {start: abs(start[0] - goal[0]) + abs(start[1] - goal[1])}

    heapq.heappush(pq, (f_score[start], 0, start))

    while pq:
        _, _, current = heapq.heappop(pq)
        if current == goal:
            break

        if current in visited:
            continue
        visited.add(current)

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue

                nxt = (current[0] + dx, current[1] + dy)
                if not is_walkable(image, nxt, obstacles):
                    continue

                step_cost = 1
                tentative = g_score[current] + step_cost
                if nxt not in g_score or tentative < g_score[nxt]:
                    came_from[nxt] = current
                    g_score[nxt] = tentative
                    est = tentative + abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                    f_score[nxt] = est
                    heapq.heappush(pq, (est, tentative, nxt))

    if goal not in came_from and goal != start:
        return []

    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


def visualize_path(image_path: str):
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'Image not found: {image_path}')

    start, goal, obstacles = detect_markers(image)
    path = find_shortest_path(image, start, goal, obstacles)

    height, width = image.shape[:2]
    output = np.full((height, width, 3), 255, dtype=np.uint8)
    cv.rectangle(output, (0, 0), (width - 1, height - 1), (0, 0, 0), 2)

    for ox, oy, ow, oh in obstacles:
        cv.rectangle(output, (ox, oy), (ox + ow, oy + oh), (128, 0, 128), -1)
        cv.rectangle(output, (ox, oy), (ox + ow, oy + oh), (0, 0, 0), 2)

    if path:
        pts = np.array(path, dtype=np.int32)
        cv.polylines(output, [pts], False, (0, 255, 255), 3)

    cv.circle(output, start, 8, (0, 255, 0), -1)
    cv.circle(output, start, 8, (0, 0, 0), 2)
    cv.circle(output, goal, 8, (0, 0, 255), -1)
    cv.circle(output, goal, 8, (0, 0, 0), 2)

    cv.putText(output, 'Output Image', (10, height - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv.LINE_AA)

    has_display = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
    if has_display:
        cv.imshow('Detected elements and path', output)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        output_path = os.path.join(os.path.dirname(image_path), 'path_result.png')
        cv.imwrite(output_path, output)
        print(f'Saved visualization to: {output_path}')
        print(f'Start: {start}, Goal: {goal}, Obstacles: {len(obstacles)}')
        if path:
            print(f'Path length: {len(path)}')


if __name__ == '__main__':
    visualize_path('./photos/1.jpg')
