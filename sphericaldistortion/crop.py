import torch
import torch.nn.functional as F
import imageio.v2 as imageio
import numpy as np

from .utils import minfocal, diskradius, interp2linear

def crop_panorama(image360, height, width, f, xi, az, el, roll):
    """Crop an image of a 360 degree panorama, given the camera parameters,

    Args:
        image360 (str or np.ndarray): image or path to image of a 360 degree panorama
        height (int): height of the cropped image
        width (int): width of the cropped image
        f (float): focal lenght of the camera in pixels
        xi (float): distortion parameter following the spherical distortion model
        az (float): azimuth of the camera in radians
        el (float): elevation of the camera in radians
        roll (float): roll of the camera in radians

    Returns:
        np.ndarray: crop of the panorama
    """

    if isinstance(image360, str):
        image360 = imageio.imread(image360) #.astype('float32') / 255.

    image360 = torch.tensor(image360.astype(float))

    u0 = width / 2.
    v0 = height / 2.

    grid_x, grid_y = np.meshgrid(list(range(width)), list(range(height)))

    ImPano_W = np.shape(image360)[1]
    ImPano_H = np.shape(image360)[0]
    x_ref = 1
    y_ref = 1

    fmin = minfocal(u0, v0, xi, x_ref, y_ref) # compute minimal focal length for the image to ve catadioptric with given xi
    
    # 1. Projection on the camera plane
    
    X_Cam = np.divide(grid_x - u0, f)
    Y_Cam = -np.divide(grid_y - v0, f)

    # 2. Projection on the sphere

    AuxVal = np.multiply(X_Cam, X_Cam) + np.multiply(Y_Cam, Y_Cam)

    alpha_cam = np.real(xi + np.sqrt(1 + np.multiply((1 - xi * xi), AuxVal)))

    alpha_div = AuxVal + 1

    alpha_cam_div = np.divide(alpha_cam, alpha_div)

    X_Sph = np.multiply(X_Cam, alpha_cam_div)
    Y_Sph = np.multiply(Y_Cam, alpha_cam_div)
    Z_Sph = alpha_cam_div - xi

    # 3. Rotation of the sphere

    coords = np.vstack((X_Sph.ravel(), Y_Sph.ravel(), Z_Sph.ravel()))
    rot_el = np.array([1., 0., 0., 0., np.cos(el), -np.sin(el), 0., np.sin(el), np.cos(el)]).reshape((3, 3))
    rot_az = np.array([np.cos(az), 0., np.sin(az), 0., 1., 0., -np.sin(az), 0., np.cos(az)]).reshape((3, 3))
    rot_roll = np.array([np.cos(roll), -np.sin(roll), 0., np.sin(roll), np.cos(roll), 0., 0., 0., 1.]).reshape((3, 3))
    sph = rot_roll.dot(rot_el.dot(coords))
    sph = rot_az.dot(sph)

    sph = sph.reshape((3, height, width)).transpose((1,2,0))
    X_Sph, Y_Sph, Z_Sph = sph[:,:,0], sph[:,:,1], sph[:,:,2]

    # 4. cart 2 sph
    ntheta = np.arctan2(X_Sph, Z_Sph)
    nphi = np.arctan2(Y_Sph, np.sqrt(Z_Sph**2 + X_Sph**2))

    pi = np.pi

    # 5. Sphere to pano
    min_theta = -pi
    max_theta = pi
    min_phi = -pi / 2.
    max_phi = pi / 2.

    min_x = 0
    max_x = ImPano_W - 1.0
    min_y = 0
    max_y = ImPano_H - 1.0

    ## for x
    a = (max_theta - min_theta) / (max_x - min_x)
    b = max_theta - a * max_x  # from y=ax+b %% -a;
    nx = (1. / a)* (ntheta - b)

    ## for y
    a = (min_phi - max_phi) / (max_y - min_y)
    b = max_phi - a * min_y  # from y=ax+b %% -a;
    ny = (1. / a)* (nphi - b)

    # 6. Final step interpolation and mapping
    image360 = image360.permute(2,0,1).unsqueeze(0)
    pano_height, pano_width = image360.shape[2:]

    # change to torch grid sample format (from -1 to 1)
    nx = torch.tensor((nx-pano_width/2)/pano_width*2)
    ny = torch.tensor((ny-pano_height/2)/pano_height*2)

    im = F.grid_sample(image360, torch.stack((nx, ny), dim=2).unsqueeze(0), mode='bilinear', padding_mode='zeros')

    im = im.squeeze().permute(1,2,0).numpy().astype(np.uint8)

    if f < fmin:  # if it is a catadioptric image, apply mask and a disk in the middle
        r = diskradius(xi, f)
        DIM = im.shape
        ci = (np.round(DIM[0]/2), np.round(DIM[1]/2))
        xx, yy = np.meshgrid(list(range(DIM[0])) - ci[0], list(range(DIM[1])) - ci[1])
        mask = np.double((np.multiply(xx,xx) + np.multiply(yy,yy)) < r*r)
        mask_3channel = np.stack([mask, mask, mask], axis=-1).transpose((1,0,2))
        im = np.array(np.multiply(im, mask_3channel), dtype=np.uint8)

    return im