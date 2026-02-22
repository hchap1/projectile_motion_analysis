import os
import re

def load_records():
    records = []

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
            launch_angle = float(lines[3].split(":")[1])

            records.append((hood, launch_angle, rpm, vel))
        except:
            continue

    return records


def print_csv(title, headers, data):
    print("\n" + title)
    print(",".join(headers))
    for row in data:
        print(",".join(f"{value:.10f}" for value in row))


def main():
    records = load_records()

    if not records:
        print("No records found.")
        return

    # Sort for cleaner tables
    hood_to_angle = sorted((hood, launch_angle) for hood, launch_angle, _, _ in records)
    rpm_to_velocity = sorted((rpm, vel) for _, _, rpm, vel in records)

    print("\nCOPY EACH SECTION INTO A SEPARATE DESMOS TABLE\n")

    print_csv(
        "Hood Angle → Launch Angle",
        ["Hood_Angle", "Launch_Angle"],
        hood_to_angle
    )

    print_csv(
        "RPM → Launch Velocity",
        ["RPM", "Launch_Velocity"],
        rpm_to_velocity
    )


if __name__ == "__main__":
    main()
