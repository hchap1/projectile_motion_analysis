import pygame
import math
import sys

pygame.init()

WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# -----------------------
# Physics parameters
# -----------------------
g = 980.0  # gravity (upwards positive)

launch_pos = (150, 500)
target_pos = (850, 350)

ideal_angle_deg = 75
ideal_angle = math.radians(ideal_angle_deg)

# Convert to physics space (origin at launch)
dx = target_pos[0] - launch_pos[0]
dy = -(target_pos[1] - launch_pos[1])  # invert for physics space


# -----------------------
# Solve velocity for fixed angle
# -----------------------
def velocity_for_angle(x, y, theta):
    cos = math.cos(theta)
    tan = math.tan(theta)

    denom = 2 * cos * cos * (x * tan - y)

    v2 = (g * x * x) / denom

    if v2 <= 0:
        raise ValueError("No physical solution")

    return math.sqrt(v2)


# -----------------------
# Solve angle for fixed velocity
# -----------------------
def angle_for_velocity(x, y, v):
    disc = v**4 - g * (g * x**2 + 2 * y * v**2)

    if disc < 0:
        raise ValueError("Target unreachable at this velocity")

    sqrt_disc = math.sqrt(disc)

    tan1 = (v**2 + sqrt_disc) / (g * x)
    tan2 = (v**2 - sqrt_disc) / (g * x)

    theta1 = math.atan(tan1)
    theta2 = math.atan(tan2)

    # choose upward solutions only
    candidates = [theta1, theta2]
    candidates = [t for t in candidates if t > 0]

    if not candidates:
        raise ValueError("No upward solution")

    return max(candidates)  # high arc

# -----------------------
# Trajectory simulation
# -----------------------
def simulate(v, theta):
    points = []
    t = 0
    dt = 0.01

    while t < 5:
        x = v * math.cos(theta) * t
        y = v * math.sin(theta) * t + 0.5 * g * t * t

        screen_x = launch_pos[0] + x
        screen_y = launch_pos[1] - y

        points.append((screen_x, screen_y))

        if screen_y > HEIGHT:
            break

        t += dt

    return points


# -----------------------
# Compute shots
# -----------------------
ideal_velocity = velocity_for_angle(dx, dy, ideal_angle)

shots = []

# Shot 1: ideal
shots.append((ideal_velocity, ideal_angle))

# Shot 2: 5% slower
v2 = ideal_velocity * 0.95
theta2 = angle_for_velocity(dx, dy, v2)
shots.append((v2, theta2))

# Shot 3: 10% slower
v3 = ideal_velocity * 0.90
theta3 = angle_for_velocity(dx, dy, v3)
shots.append((v3, theta3))

trajectories = [simulate(v, theta) for v, theta in shots]


# -----------------------
# Main loop
# -----------------------
running = True
while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((30, 30, 30))

    # Draw launch
    pygame.draw.circle(screen, (0, 255, 0), launch_pos, 8)

    # Draw target
    pygame.draw.circle(screen, (255, 0, 0), target_pos, 8)

    # Draw trajectories
    colors = [(255, 255, 0), (0, 200, 255), (255, 100, 255)]

    for i, traj in enumerate(trajectories):
        if len(traj) > 1:
            pygame.draw.lines(screen, colors[i], False, traj, 3)

    pygame.display.flip()

pygame.quit()
sys.exit()
