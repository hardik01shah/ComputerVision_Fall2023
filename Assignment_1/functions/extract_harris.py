import numpy as np
import cv2

from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # 1. Compute image gradients in x and y direction
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    dx = np.array(([[0., 0., 0.],
                    [0.5, 0., -0.5],
                    [0., 0., 0.]]))
    dy = np.array(([[0., 0.5, 0.],
                    [0., 0., 0.],
                    [0., -0.5, 0.]]))
    Ix = signal.convolve2d(img, dx, mode='same', boundary='symm')
    Iy = signal.convolve2d(img, dy, mode='same', boundary='symm')
    
    
    # 2. Blur the computed gradients
    # TODO: compute the blurred image gradients
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    # Gaussian filter can be added here to smoothen out image derivatives.
    # Ix_b = cv2.GaussianBlur(Ix, ksize=(0,0), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    # Iy_b = cv2.GaussianBlur(Iy, ksize=(0,0), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    Ix_b = Ix
    Iy_b = Iy

    # 3. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here
    Ix2 = np.multiply(Ix_b, Ix_b)
    Iy2 = np.multiply(Iy_b, Iy_b)
    Ix_Iy = np.multiply(Ix_b, Iy_b)

    # Add gaussian blur to the computed matrix elements of the local auto-correlation matrix "M"
    Ix2 = cv2.GaussianBlur(Ix2, ksize=(0,0), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    Iy2 = cv2.GaussianBlur(Iy2, ksize=(0,0), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    Ix_Iy = cv2.GaussianBlur(Ix_Iy, ksize=(0,0), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)


    # 4. Compute Harris response function C
    # TODO: compute the Harris response function C here
    C = (np.multiply(Ix2, Iy2) - np.square(Ix_Iy)) - k*np.square(Ix2 + Iy2)

    # 5. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition; Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format
    Cmax = ndimage.maximum_filter(C, 3)
    idx = np.where(np.logical_and(C>thresh, Cmax==C))
    corners = np.stack((idx[1], idx[0])).T

    return corners, C

