# homography_setup.py

import cv2
import numpy as np


class HomographyPlane:
    """
    Handles homography from image space -> real-world mm plane.
    """

    def __init__(self, width_mm, height_mm):
        self.width_mm = float(width_mm)
        self.height_mm = float(height_mm)
        self.H = None

        # Pre-build destination points once
        self._dst = np.array([
            [0, self.height_mm],           # Bottom Left
            [self.width_mm, self.height_mm],  # Bottom Right
            [self.width_mm, 0],            # Top Right
            [0, 0]                         # Top Left
        ], dtype=np.float32)

    def calibrate_from_camera(self, cap):
        """
        Click:
            Bottom Left
            Bottom Right
            Top Right
            Top Left
        Press ENTER when done.
        """

        prompts = ["Bottom Left", "Bottom Right", "Top Right", "Top Left"]
        points = []
        idx = 0

        def mouse(event, x, y, flags, param):
            nonlocal idx
            if event == cv2.EVENT_LBUTTONDOWN and idx < 4:
                points.append([x, y])
                idx += 1

        cv2.namedWindow("Calibrate Plane")
        cv2.setMouseCallback("Calibrate Plane", mouse)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()

            for p in points:
                cv2.circle(display, tuple(p), 6, (0, 255, 0), -1)

            if idx < 4:
                text = f"Click: {prompts[idx]}"
            else:
                text = "Press ENTER"

            cv2.putText(display, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Calibrate Plane", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 13 and idx == 4:
                break

        cv2.destroyWindow("Calibrate Plane")

        src = np.array(points, dtype=np.float32)
        self.H = cv2.getPerspectiveTransform(src, self._dst)

    def warp(self, frame):
        """
        Fast warp to real-world mm coordinate plane.
        Output resolution = width_mm x height_mm pixels.
        """
        return cv2.warpPerspective(
            frame,
            self.H,
            (int(self.width_mm), int(self.height_mm)),
            flags=cv2.INTER_LINEAR
        )

    def image_to_world(self, pts):
        """
        Convert Nx2 image points to mm coordinates.
        pts: np.float32 shape (N,2)
        """
        pts = pts.reshape(-1, 1, 2)
        real = cv2.perspectiveTransform(pts, self.H)
        return real.reshape(-1, 2)
