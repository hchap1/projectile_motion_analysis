import re
import os
import cv2
import numpy as np
import threading
from time import perf_counter_ns
from homography_setup import HomographyPlane

# ---------------------------------------------------------------------------
# HSV detection range (green ball)
# ---------------------------------------------------------------------------
lower = np.array([35, 20, 20])
upper = np.array([85, 255, 255])

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

# Plane dimensions — must match calibration
PLANE_W = 3000.0      # mm
PLANE_H = 1500.0      # mm  (y=0 top, y=1500 bottom/launcher)


# ---------------------------------------------------------------------------
# Aerodynamic simulation
# ---------------------------------------------------------------------------
def simulate_trajectory(vx0, vy0, x0=0.0, y0=PLANE_H,
                         drag=False, magnus=False, rpm=0.0,
                         dt=0.0005, max_t=2.0):
    """
    Integrate equations of motion in image-space mm coordinates.
      x increases right, y increases DOWN (y=0 top, y=PLANE_H bottom).
      Gravity acts in +y direction.
      Ball launched from (x0, y0) with velocity (vx0, vy0).
      vy0 < 0 means the ball is moving upward.
      Backspin for a rightward ball: Magnus force in -y direction (upward).

    Returns list of (x_mm, y_mm).
    """
    x, y   = float(x0), float(y0)
    vx, vy = float(vx0), float(vy0)
    omega  = rpm * 2.0 * np.pi / 60.0   # rad/s

    traj = [(x, y)]
    t    = 0.0

    while t < max_t:
        v_mm  = np.sqrt(vx**2 + vy**2)           # mm/s
        v_m   = v_mm / 1000.0                      # m/s  (needed for Cl/Cd calcs)

        ax = 0.0
        ay = G_MM   # gravity always pulls down (+y)

        if drag and v_mm > 1e-6:
            # Fd = 0.5 * Cd * rho * A * v²  [kg·mm/s² = mN... but rho in kg/mm³, A in mm², v in mm/s]
            # Units: kg/mm³ * mm² * (mm/s)² = kg/s² * mm = kg·mm/s²  ✓  divide by m(kg) -> mm/s²
            Fd_accel = 0.5 * CD * RHO * BALL_A * v_mm**2 / BALL_M   # mm/s²
            # Opposes velocity
            ax -= Fd_accel * vx / v_mm
            ay -= Fd_accel * vy / v_mm

        if magnus and v_m > 1e-6 and omega > 0:
            S       = omega * (BALL_R / 1000.0) / v_m          # spin ratio (dimensionless)
            Cl      = min(0.35 * S, 0.5)                        # empirical Magnus coefficient
            Fl_accel = 0.5 * Cl * RHO * BALL_A * v_mm**2 / BALL_M  # mm/s²
            # Backspin direction (verified): force = Fl * (vy/v, -vx/v)
            # For rightward ball (vx>0): fy = -Fl*vx/v < 0  -> upward ✓
            ax += Fl_accel * ( vy / v_mm)
            ay += Fl_accel * (-vx / v_mm)

        vx += ax * dt
        vy += ay * dt
        x  += vx * dt
        y  += vy * dt
        t  += dt

        traj.append((x, y))

        # Stop if ball leaves the extended plane area
        if x > PLANE_W * 1.2 or x < -PLANE_W * 0.2:
            break
        if y > PLANE_H * 1.2 or y < -PLANE_H * 0.5:
            break

    return traj


# ---------------------------------------------------------------------------
# Least-squares trajectory fit
# ---------------------------------------------------------------------------
def fit_trajectory(points):
    """
    Fit projectile equations to all detected points using least squares.

      x(t) = x0 + vx * t                        (linear in x0, vx)
      y(t) = y0 + vy * t + 0.5 * G_MM * t²      (linear in y0, vy; +g because y↓)

    points: list of (cx_mm, cy_mm, t_s, cx_px, cy_px)

    Returns: x0, y0, vx, vy  — all in mm and mm/s
             launch_speed (mm/s), launch_angle (degrees, positive = above horizontal)
    """
    n = len(points)
    ts = np.array([p[2] for p in points])
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    # X fit: x = x0 + vx*t  ->  [1, t] * [x0, vx]^T = x
    Ax = np.column_stack([np.ones(n), ts])
    res_x, _, _, _ = np.linalg.lstsq(Ax, xs, rcond=None)
    x0_fit, vx_fit = res_x

    # Y fit: y = y0 + vy*t + 0.5*g*t²  ->  [1, t] * [y0, vy]^T = y - 0.5*g*t²
    y_corrected = ys - 0.5 * G_MM * ts**2
    Ay = np.column_stack([np.ones(n), ts])
    res_y, _, _, _ = np.linalg.lstsq(Ay, y_corrected, rcond=None)
    y0_fit, vy_fit = res_y

    launch_speed = np.sqrt(vx_fit**2 + vy_fit**2)
    # vy_fit < 0 means upward in image coords; angle above horizontal
    launch_angle_deg = np.degrees(np.arctan2(-vy_fit, vx_fit))

    # Residuals for reporting
    t_vals = ts
    x_pred = x0_fit + vx_fit * t_vals
    y_pred = y0_fit + vy_fit * t_vals + 0.5 * G_MM * t_vals**2
    x_resid = xs - x_pred
    y_resid = ys - y_pred
    rmse = np.sqrt(np.mean(x_resid**2 + y_resid**2))

    return x0_fit, y0_fit, vx_fit, vy_fit, launch_speed, launch_angle_deg, rmse


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
    """
    Detect ball centroid in warped image, convert to real-world mm via H inverse.
    Returns list of (cx_mm, cy_mm, t_s, cx_px, cy_px).

    Dimensional notes:
      - plane.warp() maps raw camera -> mm plane, 1 px = 1 mm, no margin.
      - Centroid in warped image (cx_px, cy_px) == real-world (cx_mm, cy_mm) directly.
      - We nonetheless go through H_inv -> image_to_world for full rigour.
      - Timestamps are from perf_counter_ns() at cap.grab() — hardware-synchronised.
    """
    points          = []
    started         = False
    start_time_s    = None
    frames_since_last = 0
    kernel          = np.ones((5, 5), np.uint8)
    H_inv           = np.linalg.inv(plane.H)

    for raw_frame, warped, timestamp_ns in recorded:
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
            if area > 500:
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
                        start_time_s = timestamp_ns / 1e9
                    frames_since_last = 0
                    t_s    = (timestamp_ns / 1e9) - start_time_s
                    center = (cx_mm, cy_mm, t_s, int(cx_px), int(cy_px))
                    points.append(center)

        if center is None:
            frames_since_last += 1
        if started and frames_since_last > 5:
            break

    return points


# ---------------------------------------------------------------------------
# Debug replay
# ---------------------------------------------------------------------------
def debug_replay(recorded):
    kernel = np.ones((5, 5), np.uint8)
    print(f"\nDEBUG REPLAY: {len(recorded)} frames. Any key = next, ESC = done.\n")
    for i, (raw_frame, warped, ts_ns) in enumerate(recorded):
        hsv  = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0
        cx, cy = -1, -1
        if contours:
            lg = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(lg)
            if largest_area > 0:
                M = cv2.moments(lg)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
        ann = warped.copy()
        cv2.putText(ann, f"Frame {i}  t={ts_ns/1e9:.4f}s  area={largest_area:.0f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        if cx >= 0:
            cv2.circle(ann, (cx, cy), 15, (0, 0, 255), 3)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.resize(mask_bgr, (ann.shape[1], ann.shape[0]))
        combined = np.hstack([ann, mask_bgr])
        if combined.shape[1] > 1400:
            sc = 1400 / combined.shape[1]
            combined = cv2.resize(combined,
                                  (1400, int(combined.shape[0] * sc)))
        cv2.imshow("Debug (warped | mask)", combined)
        if cv2.waitKey(0) & 0xFF == 27:
            break
    cv2.destroyWindow("Debug (warped | mask)")
    cv2.waitKey(1)
    print("Debug replay done.\n")


# ---------------------------------------------------------------------------
# Trajectory display
# ---------------------------------------------------------------------------
def draw_traj_on_image(img, traj_mm, color, thickness=2):
    """Draw a list of (x_mm, y_mm) points onto the warped image."""
    pts = [(int(round(x)), int(round(y))) for x, y in traj_mm
           if 0 <= x <= PLANE_W and 0 <= y <= PLANE_H]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def show_trajectory(recorded, points, vx_fit, vy_fit, rpm):
    """
    Display last warped frame with:
      - Detected points, each labelled with real-world mm coords
      - Three theoretical arcs from (0, PLANE_H):
          1. No drag / no Magnus   (cyan)
          2. Drag only             (orange)
          3. Drag + Magnus         (magenta)
    """
    if not recorded:
        return

    display = recorded[-1][1].copy()   # last warped frame

    # --- Theoretical arcs ---
    arc_nodrag  = simulate_trajectory(vx_fit, vy_fit,
                                       drag=False, magnus=False)
    arc_drag    = simulate_trajectory(vx_fit, vy_fit,
                                       drag=True,  magnus=False)
    arc_magnus  = simulate_trajectory(vx_fit, vy_fit,
                                       drag=True,  magnus=True, rpm=rpm)

    draw_traj_on_image(display, arc_nodrag, color=(255, 255,   0), thickness=2)  # cyan
    draw_traj_on_image(display, arc_drag,   color=(  0, 165, 255), thickness=2)  # orange
    draw_traj_on_image(display, arc_magnus, color=(255,   0, 255), thickness=2)  # magenta

    # --- Legend (top-left) ---
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.55
    fthick = 1
    legend = [
        ("No drag / no Magnus", (255, 255,   0)),
        ("Drag only",            (  0, 165, 255)),
        ("Drag + Magnus",        (255,   0, 255)),
        ("Measured points",      (255, 255, 255)),
    ]
    for li, (label, col) in enumerate(legend):
        cv2.putText(display, label, (12, 30 + li * 22),
                    font, fscale, (0, 0, 0), fthick + 2, cv2.LINE_AA)
        cv2.putText(display, label, (12, 30 + li * 22),
                    font, fscale, col,       fthick,     cv2.LINE_AA)

    # --- Detected points ---
    for pt in points:
        cx_px, cy_px = pt[3], pt[4]
        cx_mm, cy_mm = pt[0], pt[1]

        cv2.circle(display, (cx_px, cy_px), 10, (255, 255, 255), -1)

        label = f"({cx_mm:.0f}, {cy_mm:.0f}) mm"
        fs    = 0.55
        ft    = 1
        (tw, th), bl = cv2.getTextSize(label, font, fs, ft)
        tx = cx_px - tw // 2
        ty = cy_px - 18
        ty = max(ty, th + 4)
        # Dark background for readability
        cv2.rectangle(display, (tx - 3, ty - th - 3),
                      (tx + tw + 3, ty + bl + 2), (0, 0, 0), -1)
        cv2.putText(display, label, (tx, ty), font, fs,
                    (255, 255, 255), ft, cv2.LINE_AA)

    clicked = {"done": False}
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["done"] = True

    cv2.namedWindow("Trajectory", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Trajectory", min(PLANE_W, 1400), min(PLANE_H, 700))
    cv2.setMouseCallback("Trajectory", on_mouse)
    cv2.imshow("Trajectory", display)
    while not clicked["done"]:
        cv2.waitKey(1)
    cv2.destroyWindow("Trajectory")
    cv2.waitKey(1)


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
        choice = input("Run debug replay? [ENTER = yes, anything else = no]: ")
        if choice.strip() == "":
            debug_replay(recorded)
        return

    # --- Least-squares fit ---
    x0_fit, y0_fit, vx_fit, vy_fit, launch_speed, launch_angle, rmse = \
        fit_trajectory(points)

    # RPM from the heuristic (1/4 of launch speed in mm/s)
    rpm = launch_speed / 4.0

    # --- Print point table ---
    print("\n--- Detected trajectory points ---")
    print(f"{'#':>3}  {'x(mm)':>8}  {'y(mm)':>8}  {'t(s)':>9}  "
          f"{'dist(mm)':>9}  {'dt(s)':>8}  {'spd(mm/s)':>10}")
    for i, pt in enumerate(points):
        if i == 0:
            print(f"{i:>3}  {pt[0]:>8.2f}  {pt[1]:>8.2f}  {pt[2]:>9.5f}")
        else:
            prev = points[i - 1]
            dx   = pt[0] - prev[0]
            dy   = pt[1] - prev[1]
            dist = np.sqrt(dx**2 + dy**2)
            dt   = pt[2] - prev[2]
            spd  = dist / dt if dt > 0 else 0
            print(f"{i:>3}  {pt[0]:>8.2f}  {pt[1]:>8.2f}  {pt[2]:>9.5f}  "
                  f"{dist:>9.2f}  {dt:>8.5f}  {spd:>10.1f}")

    print(f"\n--- Least-squares fit results ---")
    print(f"  Fitted origin:    ({x0_fit:.1f}, {y0_fit:.1f}) mm")
    print(f"  vx = {vx_fit:.2f} mm/s   vy = {vy_fit:.2f} mm/s")
    print(f"  Launch speed:     {launch_speed:.2f} mm/s  ({launch_speed/1000:.4f} m/s)")
    print(f"  Launch angle:     {launch_angle:.4f} deg (above horizontal)")
    print(f"  Fit RMSE:         {rmse:.2f} mm")
    print(f"  RPM (heuristic):  {rpm:.0f}")

    show_trajectory(recorded, points, vx_fit, vy_fit, rpm)

    rpm_input = float(input("RPM (actual, for record): "))
    hood      = float(input("HOOD angle: "))

    confirm = input("CONFIRM? [ENTER to save, anything else to exit]: ")
    if len(confirm) > 0:
        return

    record_id   = get_next_record_number()
    points_str  = "\n".join(
        f"  ({p[0]:.3f}mm, {p[1]:.3f}mm, {p[2]:.6f}s)" for p in points)
    with open(f"record_{record_id}.txt", "w") as f:
        f.write(
            f"rpm:{rpm_input}\n"
            f"hood:{hood}\n"
            f"vel:{launch_speed:.4f}\n"
            f"ang:{launch_angle:.4f}\n"
            f"vx:{vx_fit:.4f}\n"
            f"vy:{vy_fit:.4f}\n"
            f"fit_rmse:{rmse:.4f}\n"
            f"points\n{points_str}\n"
        )
    print(f"Saved to record_{record_id}.txt")


if __name__ == "__main__":
    main()
