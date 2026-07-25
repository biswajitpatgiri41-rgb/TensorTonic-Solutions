def morphological_op(image, kernel, operation):
    h, w = len(image), len(image[0])
    kh, kw = len(kernel), len(kernel[0])
    ph, pw = kh // 2, kw // 2

    padded = [[0] * (w + 2 * pw) for _ in range(h + 2 * ph)]
    for i in range(h):
        for j in range(w):
            padded[i + ph][j + pw] = image[i][j]

    output = [[0] * w for _ in range(h)]

    for i in range(h):
        for j in range(w):
            if operation == "erode":
                ok = True
                for ki in range(kh):
                    for kj in range(kw):
                        if kernel[ki][kj] == 1 and padded[i + ki][j + kj] != 1:
                            ok = False
                            break
                    if not ok:
                        break
                output[i][j] = 1 if ok else 0
            else:
                ok = False
                for ki in range(kh):
                    for kj in range(kw):
                        if kernel[ki][kj] == 1 and padded[i + ki][j + kj] == 1:
                            ok = True
                            break
                    if ok:
                        break
                output[i][j] = 1 if ok else 0

    return output