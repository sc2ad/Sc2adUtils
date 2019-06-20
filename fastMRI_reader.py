import h5py
import numpy as np
import matplotlib.pyplot as plt

def readParam_unsafe(dat, param):
    if type(dat) == str:
        dat = read_h5_unsafe(dat)
    if type(dat) == dict:
        return dat[param]
    raise NotImplementedError("Can only support strings or dicts, not: " + str(type(dat)))

def readMask(dat):
    return readParam_unsafe(dat, "mask")

def readKSpace(dat):
    return readParam_unsafe(dat, "kspace")

def read_h5_unsafe(fName):
    hf = h5py.File(fName, 'r')
    d = {k: hf[k] for k in iter(hf)}
    d['_file'] = hf
    return d

def center_crop(image, new_width=None, new_height=None):
    width = image.shape[1]
    height = image.shape[0]

    if new_width is None:
        new_width = min(width, height)
    if new_height is None:
        new_height = min(width, height)
    
    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))
    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(image.shape) == 2:
        center_cropped_image = image[top:bottom, left:right]
    else:
        center_cropped_image = image[top:bottom, left:right, ...]
    
    return center_cropped_image

def fft2c(f):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(f)))

def ifft2c(F):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F)))

def convert_to_image(kspace):
    # I don't know how to properly convert the data so that it looks correct
    # This currently just seems wrong, maybe because I am doing rfft?
    return np.array(fft2c(kspace), dtype=np.float32)

def show(image):
    plt.imshow(image)
    plt.show()