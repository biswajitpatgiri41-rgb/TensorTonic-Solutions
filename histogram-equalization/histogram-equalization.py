def histogram_equalize(image):
    hist = [0] * 256
    total_pixels = 0

    for row in image:
        for pixel in row:
            hist[pixel] += 1
            total_pixels += 1

    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    cdf_min = 0
    for value in cdf:
        if value != 0:
            cdf_min = value
            break

    if total_pixels == cdf_min:
        return [[0 for _ in row] for row in image]

    result = []
    for row in image:
        new_row = []
        for pixel in row:
            new_value = round((cdf[pixel] - cdf_min) / (total_pixels - cdf_min) * 255)
            new_row.append(new_value)
        result.append(new_row)

    return result