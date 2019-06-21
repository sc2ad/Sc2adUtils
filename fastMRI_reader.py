import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
import compressed_sensing_recon as cr
import time

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

def readImage(dat):
    return readParam_unsafe(dat, "data")

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
    return np.array(fft2c(kspace), dtype=np.float32)

def convert_to_kspace(image):
    return np.array(ifft2c(image), dtype=np.float32)

def show(image):
    plt.imshow(image)
    plt.show()

def writeImagesToDir(src, dist):
    for f in os.listdir(src):
        if f.endswith(".h5"):
            d = read_h5_unsafe(f)
            # Convert kspace to image
            img = convert_to_image(readKSpace(d))
            with open(os.path.join(dist, f) + ".pkl", 'wb') as fw:
                pkl.dump(img, fw)
            d['_file'].close()

def writeKSpacesToDir(src, dist):
    for f in os.listdir(src):
        if f.endswith(".h5"):
            d = read_h5_unsafe(f)
            # Convert image to kspace
            kspace = convert_to_kspace(readImage(d))
            with open(os.path.join(dist, f) + ".pkl", 'wb') as fw:
                pkl.dump(kspace, fw)
            d['_file'].close()

def writeTrainingRoot(src, dst, dest_pkl="dataTrainingRoot.pkl", training_percentage=0.8):
    olst = os.listdir(src)
    # Get only valid training data
    olst = [item for item in olst if item.endswith(".h5") or item.endswith(".im")]
    # only the last few are saved for validation, not exactly optimal
    lst = olst[:int(training_percentage * len(olst))]
    out = []
    for f in lst:
        start = time.time()
        label = os.path.abspath(os.path.join(src, f))
        print("Starting file: " + label)
        # Input is a new file that needs to be generated
        # Needs to be in the image space, convert to kspace, undersample,
        # then convert from kspace to image space
        d = read_h5_unsafe(label)
        # kspace = readKSpace(d)
        # image = convert_to_image(kspace)
        image = np.array(readImage(d))
        for i in range(image.shape[0]):
            print("Starting undersample for slice: " + str(i))
            image[i] = cr.image_undersampled_recon(image[i], accel_factor=12, recon_type='zero-fill')
        path = os.path.abspath(os.path.join(dst, f.replace(".im", "_undersampled.im").replace(".h5", "_undersampled.im")))
        with h5py.File(path, 'w') as fw:
            fw.create_dataset("data", image, dtype='f4')
        d['_file'].close()
        inp = path
        out.append([inp, label])
        delta = time.time() - start
        print("Completed file: " + label + " in: " + str(delta) + " seconds!")

    with open(dest_pkl, 'wb') as fw:
        pkl.dump(out, fw)
    print("Complete!")
    # Returns the files that were NOT written to the data root
    return olst[int(training_percentage * len(olst)) + 1:]
