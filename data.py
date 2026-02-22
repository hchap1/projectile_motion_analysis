import pygame
import math
import sys
import os

G = 9800.0  # mm/s^2


def load_record(number):
    filename = f"record_{number}.txt"
    if not os.path.exists(filename):
        print(f"File '{filename}' not found.")
        sys.exit(1)

    with open(filename, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Parse header
    rpm = float(lines[0].split(":")[1])
    hood = float(lines[1].split(":")[1])
    vel = float(lines[2].split(":")[1])
    ang = float(lines[3].split(":")[1])

    # Parse points
    points = []
    for line in lines[4:]:
        x, y = map(float, line.split())
        points.append((x, y))

    return rpm, hood, vel, ang, points


def main():
    record_number = input("Enter record number: ")
    rpm, hood, vel, ang, points = load_record(record_number)

    pygame.init()
    width, height = 1000, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Record {record_number}")

    clock = pygame.time.Clock()

    # Determine scaling
    all_x = [p[0] for p in points]
    all_y = [p[1] for p in points]

    max_x = max(all_x)
    max_y = max(all_y)

    margin = 50
    scale_x = (width - 2 * margin) / max_x if max_x != 0 else 1
    scale_y = (height - 2 * margin) / max_y if max_y != 0 else 1
    scale = min(scale_x, scale_y)

    def to_screen(x, y):
        # Flip Y so up is up in pygame
        sx = margin + x * scale
        sy = height - (margin + y * scale)
        return int(sx), int(sy)

    # Projectile state
    theta = math.radians(ang)
    vx = vel * math.cos(theta)
    vy = vel * math.sin(theta)

    x = 0.0
    y = 0.0
    trail = []

    running = True
    while running:
        dt = clock.tick(120) / 1000.0  # seconds

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Physics update
        x += vx * dt
        y += vy * dt
        vy -= G * dt

        trail.append((x, y))

        # Stop if below ground and moving downward
        if y < 0 and vy < 0:
            vy = 0

        # Draw
        screen.fill((0, 0, 0))

        # Draw recorded points
        for px, py in points:
            pygame.draw.circle(screen, (0, 255, 0), to_screen(px, py), 4)

        # Draw trail
        for tx, ty in trail:
            pygame.draw.circle(screen, (255, 0, 0), to_screen(tx, ty), 2)

        # Draw current projectile
        pygame.draw.circle(screen, (255, 255, 0), to_screen(x, y), 6)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
