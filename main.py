# main.py

import cv2
from homography_setup import HomographyPlane


def main():
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

        # ===================================
        # PROCESS WARPED IMAGE HERE
        # ===================================
        # For example:
        # gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(gray, 50, 150)
        # ===================================

        cv2.imshow("Warped Plane", warped)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
