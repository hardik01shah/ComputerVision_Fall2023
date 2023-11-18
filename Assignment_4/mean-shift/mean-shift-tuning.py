import time
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale
from tqdm import tqdm
from sklearn.cluster import KMeans

def distance(x, X):
    return np.linalg.norm(x-X, axis=1, keepdims=True)

def gaussian(dist, bandwidth):
    return np.exp(-(dist**2)/(2*bandwidth**2))

def update_point(weight, X):
    return np.sum(X*weight, axis=0)/np.sum(weight, keepdims=True)

def meanshift_step(X, bandwidth=2.5):
    for i in range(len(X)):
        cur_pt = X[i]
        distances = distance(cur_pt, X)
        weights = gaussian(distances, bandwidth)
        X[i] = update_point(weights, X)
    return X

def meanshift(X, bandwidth):
    for _ in tqdm(range(20)):
        X = meanshift_step(X, bandwidth)
    return X

scale = 0.5    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('eth.jpg'), scale, channel_axis=-1)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
B = [1,3,5,7]
for b in B:
    print(f"Running mean-shift for bandwidth={b}")
    t = time.time()
    X = meanshift(image_lab.copy(), b)
    t = time.time() - t
    print ('Elapsed time for mean-shift: {}'.format(t))

    # Load label colors and draw labels as an image
    colors = np.load('colors.npz')['colors']
    colors[colors > 1.0] = 1
    colors[colors < 0.0] = 0

    centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

    if len(centroids)>len(colors):
        labels_km = KMeans(n_clusters=len(colors), n_init=10).fit_predict((X / 4).round())
        result_image = colors[labels_km].reshape(shape)
        result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)     # resize result image to original resolution
        result_image = (result_image * 255).astype(np.uint8)
        io.imsave(f'result_{b}_kmeans.png', result_image)

    else:
        result_image = colors[labels].reshape(shape)
        result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)     # resize result image to original resolution
        result_image = (result_image * 255).astype(np.uint8)
        io.imsave(f'result_{b}.png', result_image)
