import numpy as np

def random_crop(image: np.ndarray, crop_size: int = 224, crop_y: int = None, crop_x: int = None) -> np.ndarray:
    h, w, _ = image.shape
    if crop_y is None:
        crop_y = np.random.randint(0, h - crop_size + 1)
    if crop_x is None:
        crop_x = np.random.randint(0, w - crop_size + 1)
    return image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size, :]

def random_horizontal_flip(image: np.ndarray, p: float = 0.5, flip_rand: float = None) -> np.ndarray:
    if flip_rand is None:
        flip_rand = np.random.rand()
    if flip_rand < p:
        return image[:, ::-1, :]
    return image

def max_pool2d(x: np.ndarray, kernel_size: int, stride: int) -> np.ndarray:
    b, h, w, c = x.shape
    h_out = (h - kernel_size) // stride + 1
    w_out = (w - kernel_size) // stride + 1
    return np.zeros((b, h_out, w_out, c), dtype=x.dtype)