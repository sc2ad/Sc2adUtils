import h5py
import numpy as np

def readParam(dat, param):
    if type(dat) == str:
        dat = read_h5(dat)
    if type(dat) == dict:
        return dat[param]
    raise NotImplementedError("Can only support strings or dicts, not: " + str(type(dat)))

def readMask(dat):
    return readParam(dat, "mask")

def readKSpace(dat):
    return readParam(dat, "kspace")

def read_h5(fName):
    with h5py.File(fName, 'r') as hf:
        return {k: hf[k] for k in iter(hf)}

def convert_to_image(kspace):
    assert type(kspace) == np.object
    assert kspace.dtype == np.float32

    return np.fft.rfft2(kspace)