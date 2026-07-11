from bisect import bisect_right

def calibrate_isotonic(cal_labels, cal_probs, new_probs):
    pairs = sorted(zip(cal_probs, cal_labels))
    probs = [p for p, _ in pairs]
    labels = [y for _, y in pairs]

    blocks = []
    for i, y in enumerate(labels):
        blocks.append([float(y), 1, i, i])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            w = b1[1] + b2[1]
            avg = (b1[0] * b1[1] + b2[0] * b2[1]) / w
            blocks.append([avg, w, b1[2], b2[3]])

    calibrated = [0.0] * len(labels)
    for avg, _, l, r in blocks:
        for i in range(l, r + 1):
            calibrated[i] = avg

    result = []
    for q in new_probs:
        if q <= probs[0]:
            result.append(calibrated[0])
        elif q >= probs[-1]:
            result.append(calibrated[-1])
        else:
            j = bisect_right(probs, q) - 1
            p1, p2 = probs[j], probs[j + 1]
            c1, c2 = calibrated[j], calibrated[j + 1]
            if p2 == p1:
                result.append(c1)
            else:
                t = (q - p1) / (p2 - p1)
                result.append(c1 + t * (c2 - c1))
    return result