import pygame
import math
import sys
import os
import colorsys

G = 9800.0  # mm/s^2


def load_record(number):
    filename = f"record_{number}.txt"
    if not os.path.exists(filename):
        return None

    with open(filename, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    rpm = float(lines[0].split(":")[1])
    hood = float(lines[1].split(":")[1])
    vel = float(lines[2].split(":")[1])
    ang = float(lines[3].split(":")[1])

    points = []
    for line in lines[4:]:
        x, y = map(float, line.split())
        points.append((x, y))

    return rpm, hood, vel, ang, points


def generate_rpm_colors(rpms):
    """Assign each RPM a unique colour."""
    sorted_rpms = sorted(list(set(rpms)))
    n = len(sorted_rpms)

    rpm_to_color = {}

    for i, rpm in enumerate(sorted_rpms):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        rpm_to_color[rpm] = (int(r * 255), int(g * 255), int(b * 255))

    return rpm_to_color


def main():
    max_number = int(input("Enter maximum record number: "))

    records = []
    all_points = []
    rpm_values = []

    for i in range(1, max_number + 1):
        data = load_record(i)
        if data is not None:
            rpm, hood, vel, ang, points = data
            records.append((i, rpm, hood, vel, ang, points))
            all_points.extend(points)
            rpm_values.append(rpm)

    if not records:
        print("No records found.")
        sys.exit(1)

    # Assign unique colours per RPM
    rpm_to_color = generate_rpm_colors(rpm_values)

    pygame.init()

    width, height = 1300, 900
    legend_width = 250

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("All Records")

    clock = pygame.time.Clock()

    # Scaling
    all_x = [p[0] for p in all_points]
    all_y = [p[1] for p in all_points]

    max_x = max(all_x) if all_x else 1
    max_y = max(all_y) if all_y else 1

    margin = 50
    scale = min(
        (width - legend_width - 2 * margin) / max_x,
        (height - 2 * margin) / max_y
    )

    def to_screen(x, y):
        sx = margin + x * scale
        sy = height - (margin + y * scale)
        return int(sx), int(sy)

    simulations = []

    for idx, rpm, hood, vel, ang, points in records:
        theta = math.radians(ang)
        vx = vel * math.cos(theta)
        vy = vel * math.sin(theta)

        color = rpm_to_color[rpm]

        simulations.append({
            "x": 0.0,
            "y": 0.0,
            "vx": vx,
            "vy": vy,
            "points": points,
            "trail": [],
            "color": color,
            "rpm": rpm
        })

    running = True
    while running:
        dt = clock.tick(120) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        # Draw simulations
        for sim in simulations:
            sim["x"] += sim["vx"] * dt
            sim["y"] += sim["vy"] * dt
            sim["vy"] -= G * dt

            sim["trail"].append((sim["x"], sim["y"]))

            if len(sim["trail"]) > 1:
                pygame.draw.lines(
                    screen,
                    sim["color"],
                    False,
                    [to_screen(x, y) for x, y in sim["trail"]],
                    2
                )

            for px, py in sim["points"]:
                pygame.draw.circle(
                    screen,
                    sim["color"],
                    to_screen(px, py),
                    3
                )

            if 0 <= sim["x"] <= max_x and 0 <= sim["y"] <= max_y:
                pygame.draw.circle(
                    screen,
                    (255, 255, 255),
                    to_screen(sim["x"], sim["y"]),
                    5
                )

        # Draw legend background
        pygame.draw.rect(
            screen,
            (20, 20, 20),
            (width - legend_width, 0, legend_width, height)
        )

        # Draw legend entries
        font = pygame.font.SysFont(None, 24)
        y_offset = 30

        for rpm in sorted(rpm_to_color.keys()):
            color = rpm_to_color[rpm]

            pygame.draw.rect(
                screen,
                color,
                (width - legend_width + 20, y_offset, 20, 20)
            )

            text = font.render(f"{rpm} RPM", True, (255, 255, 255))
            screen.blit(text, (width - legend_width + 50, y_offset))

            y_offset += 35

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
