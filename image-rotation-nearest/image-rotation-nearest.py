import math

def rotate_image(image, angle_degrees):
    h = len(image)
    w = len(image[0])
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    theta = math.radians(angle_degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    output = [[0] * w for _ in range(h)]

    for i in range(h):
        for j in range(w):
            dy = i - cy
            dx = j - cx
            src_y = cy + dy * cos_t + dx * sin_t
            src_x = cx - dy * sin_t + dx * cos_t
            sy = round(src_y)
            sx = round(src_x)
            if 0 <= sy < h and 0 <= sx < w:
                output[i][j] = image[sy][sx]

    return output