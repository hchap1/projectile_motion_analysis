import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RPM_MIN = 2000.0
RPM_MAX = 4500.0


def load_records():
    data = []

    for filename in os.listdir("."):
        match = re.match(r"record_(\d+)\.txt", filename)
        if not match:
            continue

        with open(filename, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        try:
            rpm = float(lines[0].split(":")[1])
            hood = float(lines[1].split(":")[1])
            launch_angle = float(lines[3].split(":")[1])

            data.append((hood, launch_angle, rpm))
        except:
            continue

    return data


def main():
    records = load_records()

    if not records:
        print("No records found.")
        return

    hood_vals = []
    launch_vals = []
    rpm_vals = []

    for hood, launch_angle, rpm in records:
        hood_vals.append(hood)
        launch_vals.append(launch_angle)
        rpm_vals.append(rpm)

    hood_vals = np.array(hood_vals)
    launch_vals = np.array(launch_vals)
    rpm_vals = np.array(rpm_vals)

    # Normalize RPM for colouring
    rpm_clipped = np.clip(rpm_vals, RPM_MIN, RPM_MAX)
    rpm_normalized = (rpm_clipped - RPM_MIN) / (RPM_MAX - RPM_MIN)

    # Custom green → red colormap
    cmap = LinearSegmentedColormap.from_list(
        "green_red",
        ["green", "red"]
    )

    plt.figure()
    scatter = plt.scatter(
        hood_vals,
        launch_vals,
        c=rpm_normalized,
        cmap=cmap
    )

    plt.colorbar(scatter, label="RPM (green = 2000, red = 4500)")

    plt.xlabel("Hood Angle (deg)")
    plt.ylabel("Launch Angle (deg)")
    plt.title("Launch Angle vs Hood Angle (Coloured by RPM)")

    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
