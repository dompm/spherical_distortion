import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from .utils import compute_fov, compute_focal

def undistort(im, f, xi):
    """Apply undistortion to an image.
    
    Args:
        im (str or np.ndarray): image or path to image
        f (float): focal length of the camera in pixels
        xi (float): distortion parameter following the spherical distortion model

    Returns:
        np.ndarray: undistorted image
        """

    if isinstance(im, str):
        im = imageio.imread(im)
    
    im = torch.tensor(im.astype(float))

    height, width, _ = im.shape

    fov = compute_fov(f, xi, width)

    new_xi = 0
    new_f = compute_focal(fov, new_xi, width)

    u0 = width / 2
    v0 = height / 2

    grid_x, grid_y = np.meshgrid(np.arange(0, width), np.arange(0, height))

    X_Cam = (grid_x - u0) / new_f
    Y_Cam = (grid_y - v0) / new_f

    omega = (new_xi + np.sqrt(1 + (1 - new_xi**2) * (X_Cam**2 + Y_Cam**2))) / (X_Cam**2 + Y_Cam**2 + 1)

    X_Sph = X_Cam * omega
    Y_Sph = Y_Cam * omega
    Z_Sph = omega - new_xi

    nx = X_Sph * f / (xi * np.sqrt(X_Sph**2 + Y_Sph**2 + Z_Sph**2) + Z_Sph) + u0
    ny = Y_Sph * f / (xi * np.sqrt(X_Sph**2 + Y_Sph**2 + Z_Sph**2) + Z_Sph) + v0

    im = im.permute(2,0,1).unsqueeze(0)

    # change to torch grid sample format (from -1 to 1)
    nx = torch.tensor((nx-width/2)/width*2)
    ny = torch.tensor((ny-height/2)/height*2)
    
    undistorted_im = F.grid_sample(im, torch.stack((nx, ny), dim=2).unsqueeze(0), mode='bilinear', padding_mode='zeros')

    return undistorted_im.squeeze().permute(1,2,0).numpy().astype(np.uint8)