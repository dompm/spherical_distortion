import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from .utils import compute_fov, compute_focal

def distort(im, f, xi):
    """Apply distortion to an image.

    Args:
        im (str or np.ndarray): image or path to image
        f (float): focal length of the camera in pixels
        xi (float): distortion parameter following the spherical distortion model

    Returns:
        np.ndarray: distorted image
    """
    
    if isinstance(im, str):
        im = imageio.imread(im)
    
    im = torch.tensor(im.astype(float))

    height, width, _ = im.shape

    fov = compute_fov(f, 0, width)

    new_xi = xi
    new_f = compute_focal(fov, new_xi, width)

    u0 = width / 2
    v0 = height / 2

    grid_x, grid_y = np.meshgrid(np.arange(0, width), np.arange(0, height))

    X_Cam = (grid_x - u0) / new_f
    Y_Cam = (grid_y - v0) / new_f

    omega = (new_xi + np.sqrt(1 + (1 - new_xi**2) * (X_Cam**2 + Y_Cam**2))) / (X_Cam**2 + Y_Cam**2 + 1)

    X_Sph = np.multiply(X_Cam, omega)
    Y_Sph = np.multiply(Y_Cam, omega)
    Z_Sph = omega - xi

    X_d = X_Sph*f/Z_Sph + u0
    Y_d = Y_Sph*f/Z_Sph + v0

    im = im.permute(2,0,1).unsqueeze(0)

    # change to torch grid sample format (from -1 to 1)
    X_d = torch.tensor((X_d-width/2)/width*2)
    Y_d = torch.tensor((Y_d-height/2)/height*2)

    distorted_im = F.grid_sample(im, torch.stack((X_d, Y_d), dim=2).unsqueeze(0), mode='bilinear', padding_mode='zeros')

    return distorted_im.squeeze().permute(1,2,0).numpy().astype(np.uint8)