import numpy as np

def derive_matrix_A(params):

    # no motion model
    if params['model'] == 0:
        A = np.eye(2)
    
    # constant velocity model
    elif params['model'] == 1:
        A = np.eye(4)
        A[2,0] = 1
        A[3,1] = 1
    
    else:
        raise Exception('Invalid model parameter')
    
    return A

def propagate(particles, frame_height, frame_width, params):
    
    A = derive_matrix_A(params)
    particles = np.matmul(particles, A)
    noise_position = np.random.randn(particles.shape[0], 2) * params['sigma_position']

    if params['model'] == 1:
        noise_velocity = np.random.randn(particles.shape[0], 2) * params['sigma_velocity']
        noise = np.hstack([noise_position, noise_velocity])
    else:
        noise = noise_position

    particles += noise

    # wrap particles around
    particles[:,0] = np.clip(particles[:,0], 0, frame_width-1)
    particles[:,1] = np.clip(particles[:,1], 0, frame_height-1)

    return particles