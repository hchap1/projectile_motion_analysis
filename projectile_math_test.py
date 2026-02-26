import pygame
from math import cos, radians, sin, sqrt, atan, degrees, tan, isfinite
from random import randint

_ = pygame.init()
running = True
clock = pygame.time.Clock()
screen = pygame.display.set_mode((800, 800))
font = pygame.font.Font("freesansbold.ttf", 25)

tx, ty = (3900, 1200)
G = 9800
DT = 0.01

def cost(x: tuple[float, float]) -> float:
    velocity = x[0] / 9000
    angle = x[1] / 20
    return velocity + angle

def solve_both(tx: float, ty: float) -> float | None:

    valid_solutions: list[tuple[float, float]] = []
    
    for angle in range(37, 57):
        velocity = solve_velocity(radians(angle), tx, ty, G)
        if velocity is None: continue
        if velocity > 9500: continue

        angle_a = degrees(solve_angle(velocity))
        angle_b = degrees(solve_angle(velocity * 0.95))
        angle_c = degrees(solve_angle(velocity * 0.92))

        if angle_a < 58 and angle_b < 58 and angle_c < 58:
            if angle_a > 34 and angle_b > 34 and angle_c > 34:
                _ = valid_solutions.append((velocity, abs(angle_a - angle_c)))

    if len(valid_solutions) == 0: return 10000
    valid_solutions.sort(key = lambda x: cost(x))
    return valid_solutions[0][0]


def screenify(point: tuple[float, float]) -> tuple[float, float]:
    w = screen.get_width()
    h = screen.get_height()
    return (point[0] / 5000 * w, h - point[1] / 5000 * h)

def solve_velocity(angle: float, tx: float, ty: float, G: float) -> float | None:
    cos_a = cos(angle)

    # Avoid vertical launch
    if abs(cos_a) < 1e-9:
        return None

    term = tx * tan(angle) - ty

    # Must be positive for real solution
    if term <= 1e-9:
        return None

    v_squared = (G * tx**2) / (2 * cos_a**2 * term)

    if not isfinite(v_squared) or v_squared <= 0:
        return None

    return sqrt(v_squared)

def solve_angle(v: float) -> float:

    if v == 0:
        return radians(90)

    chunk = (G * tx ** 2) / (2 * v ** 2)
    dscrm = tx ** 2 - 4 * chunk * (chunk + ty)

    if dscrm < 0:
        return radians(90)

    plus  = atan((tx + sqrt(dscrm)) / (2 * chunk))
    minus = atan((tx - sqrt(dscrm)) / (2 * chunk))

    return min(plus, minus)
    
def draw_trajectory(v: float, c: tuple[float, float, float], t: float):
    global screen

    a = solve_angle(v)
        
    if a < radians(33) or a > radians(57): return

    x, y = (0, 0)
    vx = cos(a) * v
    vy = sin(a) * v

    points: list[tuple[float, float]] = []

    while y >= 0:
        x += vx * DT
        vy -= G * DT
        y += vy * DT

        _ = points.append((x, y))

    for idx in range(1, len(points)):
        a = screenify(points[idx - 1])
        b = screenify(points[idx])
        _ = pygame.draw.line(screen, c, a, b, 3)

while running:
    _ = clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    mx, my = pygame.mouse.get_pos()
    tx = mx / 800 * 5000
    ty = 1100

    if tx < 1000: tx = 1000

    launch_velocity = solve_both(tx, ty)
    if launch_velocity is None: launch_velocity = 10000

    _ = screen.fill((0, 0, 0))
    draw_trajectory(launch_velocity, (100, 0, 0), 0)
    draw_trajectory(launch_velocity + randint(-100, 100), (200, 0, 0), 0)
    draw_trajectory(launch_velocity * 0.95, (0, 100, 0), 1)
    draw_trajectory(launch_velocity * 0.92, (0, 0, 100), 2)
    _ = screen.blit(font.render(f"LV: {round(launch_velocity)}, T: {(round(tx), round(ty))}", True, (255, 255, 255)), (10, 750))
    _ = pygame.draw.circle(screen, (255, 255, 255), (tx, ty), 20, 5)
    pygame.display.update()
pygame.quit()
