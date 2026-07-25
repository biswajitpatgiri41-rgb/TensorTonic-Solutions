def conv2d(image, kernel, stride=1, padding=0):
    h = len(image)
    w = len(image[0])
    kh = len(kernel)
    kw = len(kernel[0])

    ph = h + 2 * padding
    pw = w + 2 * padding

    padded = [[0] * pw for _ in range(ph)]

    for i in range(h):
        for j in range(w):
            padded[i + padding][j + padding] = image[i][j]

    out_h = ((h + 2 * padding - kh) // stride) + 1
    out_w = ((w + 2 * padding - kw) // stride) + 1

    output = []

    for i in range(out_h):
        row = []
        for j in range(out_w):
            s = 0.0
            for m in range(kh):
                for n in range(kw):
                    s += padded[i * stride + m][j * stride + n] * kernel[m][n]
            row.append(float(s))
        output.append(row)

    return output