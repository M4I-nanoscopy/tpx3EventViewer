import os

import h5py
import numpy as np


def psf_2d(lam, x0, y0, X, Y):
    # PSFg(r) of McMullan 2009 Eq 10 rewritten to 2D instead of radius
    return 1/(np.pi*lam**2)*np.exp(-((X-x0)**2/(lam**2)+((Y-y0)**2/(lam**2))))


def event_to_reduced_gauss(lam, x, y, edges, X, Y):
    Z = psf_2d(lam, x, y, X, Y)

    # Convert Z into whole pixels by 2D reduction along both axis
    z_reduced = np.add.reduceat(Z, edges, axis=0)
    z_reduced = np.add.reduceat(z_reduced, edges, axis=1)

    # Return normalised to 1
    return z_reduced / np.sum(z_reduced)


def generate_gaussians(lam):
    # Parameters
    delta = 0.01
    x = y = np.arange(-1, 2, delta)
    X, Y = np.meshgrid(x, y)
    edges = np.arange(0, 300, 100)

    # Final store of the matrix
    store = np.zeros((100, 100, 3, 3))

    x_positions = np.linspace(0, 1, num=100)
    y_positions = np.linspace(0, 1, num=100)

    for x_idx, x in enumerate(x_positions):
        for y_idx, y in enumerate(y_positions):
            store[y_idx, x_idx] = event_to_reduced_gauss(lam, x, y, edges, X, Y)

    return store


def get_gauss_distribution(lam):
    f = os.path.dirname(os.path.realpath(__file__))  + '/tmp/gauss-%0.2f.h5' % lam

    if os.path.exists(f):
        r = h5py.File(f, 'r')
        distribution = r['distribution'][()]
        r.close()
    else:
        print("NOTICE: Gaussian distribution with lambda %0.2f not found. Generating and saving to %s" % (lam, f))
        distribution = generate_gaussians(lam)
        w = h5py.File(f, 'w')
        w['distribution'] = distribution
        w.close()

    return distribution


def event_gaussian(distribution, shape, super_res, events):
    # Make frame bigger, to fit the Gaussian's being placed back
    extended_shape = shape * super_res + 2 * 3
    f = np.zeros((extended_shape, extended_shape))

    # Calculate super res
    events['x'] = events['x'] * super_res
    events['y'] = events['y'] * super_res

    for idx, e in enumerate(events):
        # Get the 100th index of y and x, by multiplying the float remainder by 100 and converting to int
        Z = distribution[int(e['y'] % 1 * 100), int(e['x'] % 1 * 100)]

        # We've added 3 pixels to the events, so also need to add that to base
        x_base = int(e['x']) + 3
        y_base = int(e['y']) + 3

        # Place back the gaussian
        f[y_base - 1: y_base + 2, x_base - 1: x_base + 2] += Z

    # Return only the real frame, not the extra bits
    return f[3:extended_shape+3, 3:extended_shape+3]