import math

def bilinear_resize(image, new_h, new_w):
    h = len(image)
    w = len(image[0])

    output = []

    for i in range(new_h):
        src_y = 0 if new_h == 1 else i * (h - 1) / (new_h - 1)
        y0 = int(math.floor(src_y))
        y1 = min(y0 + 1, h - 1)
        dy = src_y - y0

        row = []
        for j in range(new_w):
            src_x = 0 if new_w == 1 else j * (w - 1) / (new_w - 1)
            x0 = int(math.floor(src_x))
            x1 = min(x0 + 1, w - 1)
            dx = src_x - x0

            value = (
                image[y0][x0] * (1 - dy) * (1 - dx)
                + image[y1][x0] * dy * (1 - dx)
                + image[y0][x1] * (1 - dy) * dx
                + image[y1][x1] * dy * dx
            )

            row.append(value)
        output.append(row)

    return output