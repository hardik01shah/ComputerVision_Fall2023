import numpy as np
from chi2_cost import chi2_cost
from color_histogram import color_histogram

def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):
    
    particles_w = np.zeros([particles.shape[0], 1])
    for i in range(len(particles)):
        xmin = int(max(0, particles[i, 0] - 0.5 * bbox_width))
        ymin = int(max(0, particles[i, 1] - 0.5 * bbox_height))
        xmax = int(min(frame.shape[1]-1, particles[i, 0] + 0.5 * bbox_width))
        ymax = int(min(frame.shape[0]-1, particles[i, 1] + 0.5 * bbox_height))
        
        hist_x = color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin)

        coeff = 1./(np.sqrt(2*np.pi)*sigma_observe)
        cost2 = chi2_cost(hist_x, hist)**2
        particles_w[i,:] = coeff*np.exp(-0.5*(cost2/sigma_observe**2))
    
    particles_w = particles_w/np.sum(particles_w)

    return particles_w
