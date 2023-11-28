import numpy as np

def resample(particles, particles_w):
    ind = np.random.choice(len(particles), len(particles), replace=True, p=particles_w[:,0])
    resampled_particles = particles[ind,:]

    # chosen indices in old particle array
    indx_old = np.unique(ind)

    resampled_particles_w = np.zeros([particles.shape[0], 1]) / particles.shape[0]
    resampled_particles_w[indx_old,:] = particles_w[indx_old,:]

    assert resampled_particles.shape == particles.shape
    assert resampled_particles_w.shape == particles_w.shape

    return resampled_particles, resampled_particles_w