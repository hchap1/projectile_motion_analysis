import re
import os
import cv2
import numpy as np
import threading
from time import perf_counter_ns
from decompose_quadratic import Analysis
from homography_setup import HomographyPlane

# ---------------------------------------------------------------------------
# HSV detection range (green ball)
# ---------------------------------------------------------------------------
lower = np.array([69, 120, 38])
upper = np.array([99, 255, 255])

# ---------------------------------------------------------------------------
# Physical constants — all in mm and seconds
# ---------------------------------------------------------------------------
G_MM    = 9820.0      # gravitational acceleration  mm/s²
RHO     = 1.225e-9    # air density                 kg/mm³  (1.225 kg/m³)
BALL_M  = 0.1         # ball mass                   kg
BALL_D  = 127.0       # ball diameter               mm
BALL_R  = BALL_D / 2  # ball radius                 mm
BALL_A  = np.pi * BALL_R**2  # frontal area         mm²
CD      = 0.65        # drag coefficient (holey plastic ball)

# ---------------------------------------------------------------------------
# Camera recorder (background thread)
# ---------------------------------------------------------------------------
class CameraRecorder:
    def __init__(self, cap):
        self.cap = cap
        self._lock        = threading.Lock()
        self._frame       = None
        self._ts_ns       = None
        self._running     = False
        self._thread      = None
        self._rec_lock    = threading.Lock()
        self._recorded    = []
        self._recording   = False

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def start_recording(self):
        with self._rec_lock:
            self._recorded  = []
            self._recording = True

    def stop_recording(self):
        with self._rec_lock:
            self._recording = False

    def get_recorded(self):
        with self._rec_lock:
            return list(self._recorded)

    def latest_frame(self):
        with self._lock:
            return self._frame, self._ts_ns

    def _grab_loop(self):
        start_ns = perf_counter_ns()
        while self._running:
            grabbed = self.cap.grab()
            ts = perf_counter_ns() - start_ns   # timestamp at hardware grab
            if not grabbed:
                continue
            ret, frame = self.cap.retrieve()
            if not ret or frame is None:
                continue
            with self._lock:
                self._frame  = frame
                self._ts_ns  = ts
            with self._rec_lock:
                if self._recording:
                    self._recorded.append((frame.copy(), ts))


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
def record_video(recorder, plane):
    stop_flag = {"clicked": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            stop_flag["clicked"] = True

    cv2.namedWindow("Recording - Click to Stop")
    cv2.setMouseCallback("Recording - Click to Stop", on_mouse)
    print("Recording... Click the window to stop.")
    recorder.start_recording()

    while not stop_flag["clicked"]:
        frame, ts = recorder.latest_frame()
        if frame is None:
            cv2.waitKey(1)
            continue
        rec = recorder.get_recorded()
        n   = len(rec)
        if n >= 2:
            win = rec[max(0, n - 30):]
            dt  = (win[-1][1] - win[0][1]) / 1e9
            fps = (len(win) - 1) / dt if dt > 0 else 0
        else:
            fps = 0
        disp = frame.copy()
        cv2.putText(disp, f"Frames: {n}  FPS: {fps:.1f}  Click to stop",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Recording - Click to Stop", disp)
        cv2.waitKey(1)

    recorder.stop_recording()
    cv2.destroyWindow("Recording - Click to Stop")
    cv2.waitKey(1)

    raw = recorder.get_recorded()
    if len(raw) > 1:
        dur = (raw[-1][1] - raw[0][1]) / 1e9
        fps = (len(raw) - 1) / dur if dur > 0 else 0
        print(f"Recorded {len(raw)} frames in {dur:.3f}s  |  Measured FPS: {fps:.1f}")

    print("Warping frames...")
    warped = []
    for frame, ts in raw:
        w = plane.warp(frame)
        warped.append((frame, w, ts))
    return warped   # list of (raw, warped, timestamp_ns)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def process_frames(recorded, plane):
    points          = []
    started         = False
    frames_since_last = 0
    kernel          = np.ones((5, 5), np.uint8)
    H_inv           = np.linalg.inv(plane.H)

    for _, warped, _ in recorded:
        h_w, w_w = warped.shape[:2]

        hsv  = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        center = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area    = cv2.contourArea(largest)
            if area > 1600:
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx_px = M["m10"] / M["m00"]
                    cy_px = M["m01"] / M["m00"]

                    # Estimate ball radius in pixels from area for edge check
                    r_px = np.sqrt(area / np.pi)

                    # Discard if centroid is within one radius of any edge
                    # (partial occlusion would bias centroid)
                    if (cx_px - r_px < 0 or cx_px + r_px > w_w or
                            cy_px - r_px < 0 or cy_px + r_px > h_w):
                        if center is None:
                            frames_since_last += 1
                        continue

                    # Convert warped pixel -> raw camera pixel -> mm
                    pt_w  = np.array([[[cx_px, cy_px]]], dtype=np.float32)
                    pt_raw = cv2.perspectiveTransform(pt_w, H_inv)
                    pt_mm  = plane.image_to_world(pt_raw.reshape(-1, 2))
                    cx_mm  = float(pt_mm[0, 0])
                    cy_mm  = float(pt_mm[0, 1])

                    if not started:
                        started      = True
                    frames_since_last = 0
                    center = (cx_mm, plane.flip_y(cy_mm))
                    points.append(center)

        if center is None:
            frames_since_last += 1
        if started and frames_since_last > 5:
            break

    return points

# ---------------------------------------------------------------------------
# Record numbering
# ---------------------------------------------------------------------------
def get_next_record_number():
    pattern = re.compile(r"record_(\d+)\.txt")
    taken   = set()
    for fn in os.listdir("."):
        m = pattern.fullmatch(fn)
        if m:
            taken.add(int(m.group(1)))
    i = 0
    while i in taken:
        i += 1
    return i


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    cap.set(cv2.CAP_PROP_FPS,          60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_w   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h   = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera: {int(actual_w)}x{int(actual_h)} @ {actual_fps} FPS")

    plane = HomographyPlane()
    plane.calibrate_from_camera(cap)

    print("Ready. Press ENTER to start recording.")
    input()

    recorder = CameraRecorder(cap)
    recorder.start()
    recorded = record_video(recorder, plane)
    recorder.stop()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if len(recorded) < 3:
        print("Not enough frames recorded.")
        return

    print("Processing frames...")
    points = process_frames(recorded, plane)
    print(f"Found {len(points)} trajectory points.")

    if len(points) < 4:
        print(f"\nDetection failed (only {len(points)} points — need at least 4 for fit).")
        return

    # --- Compute launch parameters ---
    analysis = Analysis(points)
    print(analysis.launch_velocity, analysis.launch_angle_deg)

    rpm_input = float(input("RPM (actual, for record): "))
    hood      = float(input("HOOD angle: "))

    confirm = input("CONFIRM? [ENTER to save, anything else to exit]: ")
    if len(confirm) > 0:
        return

    record_id   = get_next_record_number()
    points_str  = "\n".join([f"{p[0]:.3f} {p[1]:.3f}" for p in points])

    with open(f"record_{record_id}.txt", "w") as f:
        _ = f.write(
            f"rpm:{rpm_input}\n"
            f"hood:{hood}\n"
            f"vel:{analysis.launch_velocity:.4f}\n"
            f"ang:{analysis.launch_angle_deg:.4f}\n"
            f"{points_str}\n"
        )


if __name__ == "__main__":
    main()
