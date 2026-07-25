import math

def sobel_edges(image):
    rows = len(image)
    cols = len(image[0])

    padded = [[0] * (cols + 2) for _ in range(rows + 2)]
    for i in range(rows):
        for j in range(cols):
            padded[i + 1][j + 1] = image[i][j]

    kx = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
    ky = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]

    result = [[0.0] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            gx = 0
            gy = 0
            for x in range(3):
                for y in range(3):
                    value = padded[i + x][j + y]
                    gx += value * kx[x][y]
                    gy += value * ky[x][y]
            result[i][j] = math.sqrt(gx * gx + gy * gy)

    return result