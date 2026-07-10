import numpy as np

def apply_homogeneous_transform(T, points):
    points = np.asarray(points)
    single = points.ndim == 1
    if single:
        points = points.reshape(1, 3)
    points_h = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
    transformed = (T @ points_h.T).T[:, :3]
    if single:
        return transformed.reshape(3,)
    return transformed