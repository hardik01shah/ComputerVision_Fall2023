import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please

    q1, q2 = desc1.shape[0], desc2.shape[0]

    # vectorized implementation
    idx, idy = np.meshgrid(np.arange(q2), np.arange(q1))

    distances = np.sum(np.square(desc1[idy] - desc2[idx]), axis=2)

    """
    # using for loop
    distances_loop = np.zeros((q1, q2))
    for i in np.arange(q1):
        for j in np.arange(q2):
            distances_loop[i,j] = np.sum(np.square(desc1[i,:]-desc2[j,:]))
    assert np.sum(distances_loop != distances) == 0
    """
    return distances



def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        # You may refer to np.argmin to find the index of the minimum over any axis
        matches = np.stack((np.arange(q1), np.argmin(distances, axis=1))).T

    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # You may refer to np.min to find the minimum over any axis
        match_1 = np.zeros_like(distances).astype(bool)
        match_2 = np.zeros_like(distances).astype(bool)
        match_1[(np.arange(q1), np.argmin(distances, axis=1))] = True
        match_2[(np.argmin(distances, axis=0), np.arange(q2))] = True
        idx = np.where(np.logical_and(match_1, match_2))
        matches = np.stack((idx[0], idx[1])).T

    elif method == "ratio":
        # TODO: implement the ratio test matching here
        # You may use np.partition(distances,2,axis=0)[:,1] to find the second smallest value over a row
        match_1 = np.stack((np.arange(q1), np.argmin(distances, axis=1))).T
        match_val_idx = np.where((np.min(distances, axis=1)/np.partition(distances, 2, axis=1)[:,1]) < ratio_thresh)
        matches = match_1[match_val_idx]

    else:
        raise ValueError(f"Value of method must be in [one-way, mutual, ratio]. Got {method}.")
    return matches

