import h5py
import numpy as np

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

def convert_to_image(kspace):
    assert type(kspace) == np.object
    assert kspace.dtype == np.float32

    return np.fft.rfft2(kspace)