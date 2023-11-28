import numpy as np

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    patch = frame[ymin:ymax, xmin:xmax].reshape(-1, 3)
    hist_r = np.histogram(patch[:,0], bins=hist_bin, range=(0, 255))
    hist_g = np.histogram(patch[:,1], bins=hist_bin, range=(0, 255))
    hist_b = np.histogram(patch[:,2], bins=hist_bin, range=(0, 255))

    hist = np.hstack([hist_r[0], hist_g[0], hist_b[0]])
    hist = hist / np.sum(hist)

    return hist
