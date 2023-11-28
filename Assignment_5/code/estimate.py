import numpy as np

def estimate(particles, particles_w):
    return np.sum(particles * particles_w, axis=0)