import cv2
import numpy as np
import os


class HomographyPlane:
    """
    Handles homography from image space -> real-world mm plane.
    No margins. The warped output is exactly width_mm x height_mm pixels,
    1 pixel = 1 mm. Use image_to_world() to convert raw camera coords to mm.
    """

    CACHE_FILE = "homography_calibration.npz"
    MARGIN_RATIO = 0.0  # kept for API compatibility but unused

    def __init__(self):
        self.width_mm = None
        self.height_mm = None
        self.H = None       # raw image -> mm (no margin shift)

    def _save_calibration(self):
        np.savez(
            self.CACHE_FILE,
            H=self.H,
            width_mm=self.width_mm,
            height_mm=self.height_mm
        )

    def _load_calibration(self):
        data = np.load(self.CACHE_FILE)
        self.H = data["H"]
        self.width_mm = float(data["width_mm"])
        self.height_mm = float(data["height_mm"])

    def _recalibrate(self, cap):
        self.width_mm  = float(input("Enter plane width  (mm): "))
        self.height_mm = float(input("Enter plane height (mm): "))

        # Destination corners: exact mm rectangle, no margin.
        # Origin (0,0) = bottom-left, y increases upward.
        dst = np.array([
            [0,               self.height_mm],  # Bottom Left
            [self.width_mm,   self.height_mm],  # Bottom Right
            [self.width_mm,   0             ],  # Top Right
            [0,               0             ],  # Top Left
        ], dtype=np.float32)

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
            text = f"Click: {prompts[idx]}" if idx < 4 else "Press ENTER"
            cv2.putText(display, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Calibrate Plane", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and idx == 4:
                break

        cv2.destroyWindow("Calibrate Plane")

        if len(points) != 4:
            raise RuntimeError("Calibration aborted before 4 points were selected.")

        src = np.array(points, dtype=np.float32)
        self.H = cv2.getPerspectiveTransform(src, dst)
        self._save_calibration()
        print("Calibration saved.")

    def calibrate_from_camera(self, cap):
        if os.path.exists(self.CACHE_FILE):
            choice = input(
                "Cached calibration found.\n"
                "Press ENTER to use it, or type anything to recalibrate: "
            )
            if choice.strip() == "":
                self._load_calibration()
                print("Loaded cached calibration.")
                return
            else:
                print("Recalibrating...")
                self._recalibrate(cap)
        else:
            print("No cached calibration found. Starting calibration...")
            self._recalibrate(cap)

    def warp(self, frame):
        """
        Warp raw camera frame to mm plane.
        Output is exactly width_mm x height_mm pixels — 1 px = 1 mm.
        """
        out_w = int(self.width_mm)
        out_h = int(self.height_mm)
        return cv2.warpPerspective(
            frame, self.H, (out_w, out_h),
            flags=cv2.INTER_LINEAR
        )

    def image_to_world(self, pts):
        """
        Convert Nx2 raw camera image points to mm coordinates.
        Returns Nx2 array in mm. x in [0, width_mm], y in [0, height_mm].
        """
        pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        real = cv2.perspectiveTransform(pts, self.H)
        return real.reshape(-1, 2)
