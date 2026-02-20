# main.py

from time import perf_counter_ns
from turtle import position
from pygame import Vector2
import time
import cv2
import numpy as np
from homography_setup import HomographyPlane

points = []

lower = np.array([35, 90, 90])
upper = np.array([85, 255, 255])

started = False
frames_since_last = 0

def main():
    global frames_since_last, points, started

    start_time = None
    width_mm = float(input("Plane width (mm): "))
    height_mm = float(input("Plane height (mm): "))

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    # Optional: reduce resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    plane = HomographyPlane(width_mm, height_mm)
    plane.calibrate_from_camera(cap)

    print("Running. Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        warped = plane.warp(frame)
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        filtered = cv2.bitwise_and(warped, warped, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > 50:
                M = cv2.moments(largest)

                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]

                    if not started:
                        started = True
                        start_time = perf_counter_ns()
                    frames_since_last = 0

                    center = (int(cx), int(cy), (perf_counter_ns() - start_time) / 1e9)
                    _ = points.append(center)

                else: center = None
            else: center = None
        else: center = None

        if center != None: _ = cv2.circle(filtered, (center[0], center[1]), 100, (255, 255, 255), 5)
        else: frames_since_last += 1
        for point in points:
            _ = cv2.circle(filtered, (point[0], point[1]), 100, (255, 255, 255), 5)

        cv2.imshow("Warped Plane", filtered)

        if started and frames_since_last > 5:
            break

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    rpm = float(input("RPM: "))
    hood_angle = float(input("HOOD: "))
    confirm = input("CONFIRM? [ENTER], anything else to exit.")

    if len(confirm) > 0: exit()
    
    position_one = Vector2(points[0][0], points[0][1])
    position_two = Vector2(points[1][0], points[1][1])
    position_thr = Vector2(points[2][0], points[2][1])

    time_one = points[0][2]
    time_two = points[1][2]
    time_thr = points[2][2]

    distance_a = position_two - position_one
    distance_b = position_thr - position_two

    speed_a = distance_a / (time_two - time_one)
    speed_b = distance_b / (time_thr - time_two)

    speed = (speed_a + speed_b) / 2
    print(speed)

if __name__ == "__main__":
    main()
