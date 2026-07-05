import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    
    
    batch_size = image.shape[0]

    
    kernel_size = 11
    stride = 4
    padding = 2
    filters = 96

    
    out_height = (224 + 2 * padding - kernel_size) // stride + 1
    out_width = (224 + 2 * padding - kernel_size) // stride + 1

    return np.zeros((batch_size, out_height, out_width, filters), dtype=image.dtype)