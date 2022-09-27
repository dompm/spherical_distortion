import numpy as np

# compute field of view from focal length and xi
def compute_fov(f, xi, width):
    return 2 * np.arccos((xi + np.sqrt(1 + (1 - xi**2) * (width/2/f)**2)) / ((width/2/f)**2 + 1) - xi)

# compute focal length from field of view and xi
def compute_focal(fov, xi, width):
    return width / 2 * (xi + np.cos(fov/2)) / np.sin(fov/2)

# compute the minimum focal for the image to be catadioptric given xi
def minfocal(u0, v0, xi, xref=1, yref=1):
    if -(1 - xi**2)*((xref-u0) * (xref-u0) + (yref-v0) * (yref-v0)) < 0 :
        fmin = np.nan
    else:
        fmin = np.sqrt(-(1 - xi**2)*((xref-u0) * (xref-u0) + (yref-v0) * (yref-v0)))

    return fmin * 1.0001

# compute the disk radius when the image is catadioptric
def diskradius(xi, f):
    return np.sqrt(-(f*f)/(1-xi*xi))