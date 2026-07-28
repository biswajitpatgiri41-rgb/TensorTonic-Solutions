def color_to_grayscale(image):
    grayscale = []
    for row in image:
        gray_row = []
        for pixel in row:
            r, g, b = pixel
            gray_row.append(0.299 * r + 0.587 * g + 0.114 * b)
        grayscale.append(gray_row)
    return grayscale