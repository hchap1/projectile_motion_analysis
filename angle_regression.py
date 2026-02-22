import os
import re
import numpy as np


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

            data.append((rpm, hood, launch_angle))
        except:
            continue

    return data


def main():
    records = load_records()

    if len(records) < 2:
        print("Not enough data for regression.")
        return

    # Build matrices
    X = []
    y = []

    for rpm, hood, launch_angle in records:
        X.append([rpm, hood, 1])  # bias term
        y.append(launch_angle)

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # Least squares solution
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = coeffs

    # Predictions
    y_pred = X @ coeffs

    # R^2 calculation
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Disable scientific notation
    np.set_printoptions(suppress=True)

    print("\nMultivariable Linear Regression Result:\n")

    print("Launch_Angle =")
    print(f"    {a:.10f} * RPM")
    print(f"  + {b:.10f} * Hood")
    print(f"  + {c:.10f}\n")

    print(f"R^2 = {r_squared:.10f}\n")

    print("Desmos Form:")
    print("LA = a*R + b*H + c")
    print("\nWhere:")
    print("R = RPM")
    print("H = Hood Angle")
    print("LA = Predicted Launch Angle")


if __name__ == "__main__":
    main()
