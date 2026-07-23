import numpy as np

def rotate_around_z(points, theta):
    points = np.asarray(points)
    single = points.ndim == 1
    if single:
        points = points.reshape(1, 3)

    c = np.cos(theta)
    s = np.sin(theta)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    rotated = np.empty_like(points, dtype=np.result_type(points.dtype, float))
    rotated[:, 0] = x * c - y * s
    rotated[:, 1] = x * s + y * c
    rotated[:, 2] = z

    return rotated.reshape(3,) if single else rotated