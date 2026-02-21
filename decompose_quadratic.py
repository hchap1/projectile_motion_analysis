from math import atan, cos, degrees, sqrt
import numpy as np

G = 9800

def make_quadratic(points: list[tuple[float, float]]) -> tuple[float, float, float]:

    if len(points) < 3:
        raise ValueError("At least 3 points are required for quadratic regression.")

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    X = np.column_stack((xs**2, xs, np.ones_like(xs)))
    a, b, c = np.linalg.lstsq(X, ys, rcond=None)[0]
    return (a, b, c)

def decompose(a: float, b: float, c: float):
    dscrm = sqrt(b ** 2 - 4 * a * c)
    x_start = (-b + dscrm) / (2 * a)
    x_end = (-b - dscrm) / (2 * a)

    x_apex = (x_start + x_end) / 2
    y_apex = a * x_apex ** 2 + b * x_apex + c

    # y_apex is height, therefore
    airtime = 2 * sqrt((2 * y_apex) / G)
    
    # x_velocity is constant, therefore
    x_velocity = (x_end - x_start) / airtime

    launch_slope = 2 * a * x_start + b
    launch_angle = atan(launch_slope)

    # x_velocity = velocity * cos(theta), therefore
    launch_velocity = x_velocity / cos(launch_angle)

    return (launch_velocity, launch_angle)

class Analysis:
    launch_velocity: float
    launch_angle_deg: float

    def __init__(self, points: list[tuple[float, float]]):
        a, b, c = make_quadratic(points)
        self.launch_velocity, launch_angle_rad = decompose(a, b, c)
        self.launch_angle_deg = degrees(launch_angle_rad)
