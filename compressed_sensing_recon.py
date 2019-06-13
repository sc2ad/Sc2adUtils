import numpy as np
import scipy as sp
import sigpy
import sigpy.mri
import math

def poisson_trajectory(kspace_shape, accel_factor, *args):
    """
    This function returns the poisson mask for a given kspace size and accel_factor.

    Inputs:
    :param kspace_shape: the size of the kspace to use in determining the size of the mask (tuple)
    :param accel_factor: the acceleration factor (float)
    :param args: extraneous arguments to use for sampling
    :return: the poisson disk trajectory of the input kspace_shape and acceleration factor
    """
    poisson_mask = sigpy.mri.poisson(kspace_shape, accel=accel_factor, crop_corner=True)
    # Ensures that center of k-space gets sampled
    poisson_mask[np.int16(poisson_mask.shape[0] / 2), np.int16(poisson_mask.shape[1] / 2)] = True
    return np.bool8(poisson_mask)

def reduction_disk_trajectory(kspace_shape, accel_factor, k=0.0, step=1, *args):
    """
    This function returns the reduction disk (a name I randomly came up with) for a given kspace size and accel_factor.

    Inputs:
    :param kspace_shape: the size of the kspace to use in determining the size of the mask (tuple)
    :param accel_factor: the acceleration factor (float)
    :param args: extraneous arguments to use for sampling
    :return: the reduction disk trajectory of the input kspace_shape and acceleration factor
    """
    assert step >= 1, "Circle step must always be positive!"
    assert type(k) == float, "Type of k must be a float!"
    mask = np.zeros(kspace_shape, dtype=np.bool)
    center = [np.int16(mask.shape[0] / 2), np.int16(mask.shape[1] / 2)]
    # Generate circles of increasing radii starting at the center.
    minradii = 3
    # maxradii = int(kspace_shape[0] / 2 - k)
    maxradii = int(math.sqrt(center[0] * center[0] + center[1] * center[1]))
    radii = range(minradii, maxradii, step)
    # accel_factor decrease should increase the quantity of circles
    mask[center[0], center[1]] = True
    for i in radii:
        points = i * i * math.pi / k / accel_factor # Use the circumference to determine the number of points to create
        for p in range(int(points)):
            r = np.random.random_sample() * i # Choose a random radius somewhere within the circle
            theta = np.random.random_sample() * 2 * math.pi # Choose a random theta
            x = np.int16(r * math.cos(theta) + center[0])
            y = np.int16(r * math.sin(theta) + center[1])
            if x >= kspace_shape[0] or x < 0:
                continue
            if y >= kspace_shape[1] or y < 0:
                continue
            mask[x, y] = True # Mask the point, even if it has been masked before
    return mask

def image_undersampled_recon(image, accel_factor=1.5, eps=1e-6, recon_type='l1-wavelet', trajectory=poisson_trajectory, *args):
    """
    This function undersamples the input image in k-space and performs a reconstruction with the undersampled data.

    Inputs:
    :param image: the image to be undersampled and reconstruced (numpy array complex128)
    :param accel_factor: the acceleration factor (float)
    :param eps: precision parameter for constrained reconstruction (float)
    :param recon_type: specify the reconstruction type. Accepts: 'l1-wavelet' for L1-wavelet constrained reconstruction
        (default), 'zero-fill' for a zero-filled image reconstruction
    :param trajectory: the trajectory function to use for generating the kspace mask (function)
    :param args: extraneous arguments to pass to the trajectory function call (extraneous arguments)
    :return: reconstructed_image: (numpy array complex128)
    """
    # Convert image to k-space 
    kspace = sigpy.fft(image, center=True, norm='ortho')

    # Generate trajectory mask from kspace size and accel_factor, as well as extraneous arguments.
    trajectory_mask = trajectory(kspace.shape, accel_factor, *args)

    # Generate undersampled k-space data
    undersampled_kspace = kspace * trajectory_mask

    if recon_type.lower() == 'zero-fill'.lower():
        # Reconstruct zero-filled image
        zero_filled_img = sigpy.ifft(undersampled_kspace, center=True, norm='ortho')
        return zero_filled_img

    if recon_type.lower() == 'l1-wavelet'.lower():
        # Convert poisson mask into boolean array for indexing
        mask = np.bool8(trajectory_mask)

        # Build coordinates matrix using poisson mask
        x = np.linspace(-image.shape[0] / 2, image.shape[0] / 2 - 1, image.shape[0])
        y = np.linspace(-image.shape[1] / 2, image.shape[1] / 2 - 1, image.shape[1])
        X, Y = np.meshgrid(x, y)
        x_idx = Y[mask]  # Dimensions needs to be swapped because of row-major indexing
        y_idx = X[mask]
        coords = np.concatenate([x_idx[:, np.newaxis], y_idx[:, np.newaxis]], axis=1)

        mask = np.bool8(trajectory_mask)
        L1ConstrainedWaveletApp = sigpy.mri.app.L1WaveletConstrainedRecon(y=kspace[mask],
                                                                          mps=np.ones((1, image.shape[0],
                                                                                       image.shape[1])),
                                                                          coord=coords,
                                                                          eps=eps)
        L1out = L1ConstrainedWaveletApp.run()
        return L1out
