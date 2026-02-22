import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
            vel = float(lines[2].split(":")[1])

            data.append((rpm, vel, hood))
        except:
            continue

    return data


def main():
    records = load_records()

    if not records:
        print("No records found.")
        return

    rpm_vals = np.array([r[0] for r in records])
    vel_vals = np.array([r[1] for r in records])
    hood_vals = np.array([r[2] for r in records])

    # Normalize hood for colour mapping
    hood_min = np.min(hood_vals)
    hood_max = np.max(hood_vals)

    if hood_max == hood_min:
        hood_norm = np.zeros_like(hood_vals)
    else:
        hood_norm = (hood_vals - hood_min) / (hood_max - hood_min)

    # Colormap (blue → red for clear contrast)
    cmap = LinearSegmentedColormap.from_list(
        "hood_gradient",
        ["blue", "red"]
    )

    plt.figure()
    scatter = plt.scatter(
        rpm_vals,
        vel_vals,
        c=hood_norm,
        cmap=cmap
    )

    plt.colorbar(scatter, label="Hood Angle (low → high)")

    plt.xlabel("RPM")
    plt.ylabel("Launch Velocity (mm/s)")
    plt.title("RPM vs Launch Velocity (Coloured by Hood Angle)")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
