import cv2
import numpy as np
from homography_setup import HomographyPlane

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    cap.set(cv2.CAP_PROP_FPS,          60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    plane = HomographyPlane()
    plane.calibrate_from_camera(cap)

    cv2.namedWindow("HSV Tuner")
    cv2.createTrackbar("H min", "HSV Tuner",  35,  179, lambda x: None)
    cv2.createTrackbar("H max", "HSV Tuner",  85,  179, lambda x: None)
    cv2.createTrackbar("S min", "HSV Tuner",  90,  255, lambda x: None)
    cv2.createTrackbar("S max", "HSV Tuner", 255,  255, lambda x: None)
    cv2.createTrackbar("V min", "HSV Tuner",  90,  255, lambda x: None)
    cv2.createTrackbar("V max", "HSV Tuner", 255,  255, lambda x: None)

    print("Adjust trackbars to tune HSV range. Press ESC to quit.")
    print("Current values will be printed to console on exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        warped = plane.warp(frame)
        hsv    = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("H min", "HSV Tuner")
        h_max = cv2.getTrackbarPos("H max", "HSV Tuner")
        s_min = cv2.getTrackbarPos("S min", "HSV Tuner")
        s_max = cv2.getTrackbarPos("S max", "HSV Tuner")
        v_min = cv2.getTrackbarPos("V min", "HSV Tuner")
        v_max = cv2.getTrackbarPos("V max", "HSV Tuner")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask  = cv2.inRange(hsv, lower, upper)

        # Show warped and mask side by side
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([warped, mask_bgr])

        # Scale to fit screen if needed
        if combined.shape[1] > 1400:
            scale    = 1400 / combined.shape[1]
            combined = cv2.resize(combined,
                                  (1400, int(combined.shape[0] * scale)))

        cv2.imshow("HSV Tuner", combined)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    print(f"\nlower = np.array([{h_min}, {s_min}, {v_min}])")
    print(f"upper = np.array([{h_max}, {s_max}, {v_max}])")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
