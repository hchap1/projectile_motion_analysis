import pygame
import math
import sys

from decompose_quadratic import decompose

a = -0.000699606535
b = 1.62451977
c = 85.73393

velocity, angle_rad = decompose(a, b, c)

# -------------------------------------------------
# SCREEN / PHYSICS SETTINGS (mm units)
# -------------------------------------------------
WIDTH = 3000
HEIGHT = 1500
G = 9800  # mm/s^2

SCALE = 0.2  # pixels per mm

# -------------------------------------------------
# Setup
# -------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((int(WIDTH * SCALE), int(HEIGHT * SCALE)))
pygame.display.set_caption("Projectile Simulator")

clock = pygame.time.Clock()

# -------------------------------------------------
# User Input
# -------------------------------------------------
vx = velocity * math.cos(angle_rad)
vy = -velocity * math.sin(angle_rad)  # screen y-down

# Start at bottom-left
x = 0.0
y = HEIGHT

dt = 0.001

trajectory = []

# -------------------------------------------------
# Main Loop
# -------------------------------------------------
running = True
while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Physics
    vy += G * dt
    x += vx * dt
    y += vy * dt

    trajectory.append((x, y))

    if x > WIDTH or y > HEIGHT or y < 0:
        running = False

    # -------------------------------------------------
    # Draw
    # -------------------------------------------------
    screen.fill((30, 30, 30))

    # Draw quadratic curve
    quad_points = []
    step = 5  # mm step for smoothness

    for px in range(0, WIDTH, step):
        py = a * px**2 + b * px + c
        if 0 <= py <= HEIGHT:
            quad_points.append((int(px * SCALE), HEIGHT * SCALE - int(py * SCALE)))

    if len(quad_points) > 1:
        pygame.draw.lines(screen, (0, 0, 255), False, quad_points, 2)

    # Draw trajectory
    if len(trajectory) > 1:
        points = [
            (int(px * SCALE), int(py * SCALE))
            for px, py in trajectory
        ]
        pygame.draw.lines(screen, (0, 255, 0), False, points, 2)

    # Draw projectile
    if 0 <= x <= WIDTH and 0 <= y <= HEIGHT:
        pygame.draw.circle(
            screen,
            (255, 0, 0),
            (int(x * SCALE), int(y * SCALE)),
            6
        )

    pygame.display.flip()

pygame.quit()
sys.exit()
