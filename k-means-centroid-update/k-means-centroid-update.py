def k_means_centroid_update(points, assignments, k):
    if not points:
        return []
    d = len(points[0])
    sums = [[0.0] * d for _ in range(k)]
    counts = [0] * k
    for point, cluster in zip(points, assignments):
        counts[cluster] += 1
        for i in range(d):
            sums[cluster][i] += point[i]
    centroids = []
    for cluster in range(k):
        if counts[cluster] == 0:
            centroids.append([0.0] * d)
        else:
            centroids.append([sums[cluster][i] / counts[cluster] for i in range(d)])
    return centroids