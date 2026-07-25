def image_histogram(image):
    hist = [0] * 256
    for row in image:
        for pixel in row:
            hist[pixel] += 1
    return hist