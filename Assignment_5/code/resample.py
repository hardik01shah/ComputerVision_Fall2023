import numpy as np

def resample(particles, particles_w):
    ind = np.random.choice(len(particles), len(particles), replace=True, p=particles_w[:,0])
    resampled_particles = particles[ind,:]
    resampled_particles_w = particles_w[ind,:]

    # normalize weights
    resampled_particles_w = resampled_particles_w/np.sum(resampled_particles_w)

    assert resampled_particles.shape == particles.shape
    assert resampled_particles_w.shape == particles_w.shape

    return resampled_particles, resampled_particles_w