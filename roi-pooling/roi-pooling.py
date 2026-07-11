import math

def roi_pool(feature_map, rois, output_size):
    h = len(feature_map)
    w = len(feature_map[0]) if h else 0
    pooled = []

    for x1, y1, x2, y2 in rois:
        roi_h = y2 - y1
        roi_w = x2 - x1
        out = []

        for i in range(output_size):
            row = []
            h_start = y1 + math.floor(i * roi_h / output_size)
            h_end = y1 + math.floor((i + 1) * roi_h / output_size)
            if h_end == h_start:
                h_end = h_start + 1
            h_start = max(0, min(h_start, h))
            h_end = max(h_start + 1, min(h_end, h))

            for j in range(output_size):
                w_start = x1 + math.floor(j * roi_w / output_size)
                w_end = x1 + math.floor((j + 1) * roi_w / output_size)
                if w_end == w_start:
                    w_end = w_start + 1
                w_start = max(0, min(w_start, w))
                w_end = max(w_start + 1, min(w_end, w))

                m = feature_map[h_start][w_start]
                for r in range(h_start, h_end):
                    for c in range(w_start, w_end):
                        if feature_map[r][c] > m:
                            m = feature_map[r][c]
                row.append(m)
            out.append(row)
        pooled.append(out)

    return pooled