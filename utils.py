import random
import numpy as np

import torch


def generate_points_around_line(n=64, a_range=(-5, 5), b_range=(-100, 100), noise_level=10):
    a = random.uniform(*a_range)
    b = random.uniform(*b_range)

    points = []
    for _ in range(n):
        x = random.uniform(-100, 100)
        y = a * x + b + random.uniform(-noise_level, noise_level)
        points.append((x, y))

    return points, a, b


def generate_dataset(num_samples=1024, num_points=64):
    inputs = []
    targets = []
    for _ in range(num_samples):
        points, a, b = generate_points_around_line(n=num_points)
        inputs.append([val for point in points for val in point])
        targets.append([a, b])
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


def find_best_fit_line(points):
    x_values, y_values = zip(*points)
    A = np.vstack([x_values, np.ones(len(x_values))]).T
    a, b = np.linalg.lstsq(A, y_values, rcond=None)[0]
    return a, b
