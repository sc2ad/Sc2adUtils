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

def convert_to_image(kspace):
    # I don't know how to properly convert the data so that it looks correct
    # This currently just seems wrong, maybe because I am doing rfft?
    return np.array(np.fft.rfftn(np.array(kspace)), dtype=np.float32)

def show(image):
    plt.imshow(image)
    plt.show()