import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
import compressed_sensing_recon as cr
import time
import sigpy

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

messages = {"FULL_SUCCESS": "CREATED AND COPIED", "SUCCESS_SKIP": "SUCCESSFUL ADDITION, SKIPPED CREATION", "FAILED": "FAILED DUE TO EXCEPTION"}

def createStatus(status):
    messages = {}
    return {"status": status, "message": messages.get(status, "UNKNOWN")}

def createRootKSpace(src, dst, lst, dst_pkl, unique_mask_per_slice=False, skip_existing=True, verbose=False):
    rate = 10
    accelF = 12
    out = []
    outp_result = {}
    ind = 0
    for f in lst:
        start = time.time()
        label = os.path.abspath(os.path.join(src, f))
        print("Starting file: " + label + "(" + str(ind) + "/" + str(len(lst)) + ")")
        path = os.path.abspath(os.path.join(dst, f.replace(".im", "_undersampled.im").replace(".h5", "_undersampled.im")))
        if os.path.exists(path) and skip_existing:
            print("Skipping file because it already exists!")
            out.append([path, label])
            outp_result[label] = {"status": createStatus("SUCCESS_SKIP"), "time": time.time() - start}
            ind += 1
            continue
        try:
            # Input is a new file that needs to be generated
            # Needs to be in the image space, convert to kspace, undersample,
            # then convert from kspace to image space
            d = read_h5_unsafe(label)
            # kspace = readKSpace(d)
            # image = convert_to_image(kspace)
            image = np.array(readImage(d))
            if verbose:
                print("Shape: " + str(image.shape))
            if not unique_mask_per_slice:
                mask = cr.poisson_trajectory(image[:, :, 0].shape, accelF)
            for i in range(image.shape[2]):
                if not i % rate and verbose:
                    print("Starting undersample for slice: " + str(i))
                if unique_mask_per_slice:
                    image[:, :, i] = cr.image_undersampled_recon(image[:, :, i], accel_factor=accelF, recon_type='zero-fill')
                else:
                    kspace = sigpy.fft(image[:, :, i], center=True, norm='ortho')
                    image[:, :, i] = sigpy.ifft(kspace * mask, center=True, norm='ortho')
            with h5py.File(path, 'w') as fw:
                # fw.create_dataset("data", image.shape, dtype='f4')
                fw['data'] = image
            d['_file'].close()
            out.append([path, label])
            delta = time.time() - start
            print("Completed file: " + label + " in: " + str(delta) + " seconds!")
            outp_result[label] = {"status": createStatus("FULL_SUCCESS"), "time": delta}
        except:
            print("Failed to create file at path: " + path + " from label: " + label + "!")
            outp_result[label] = {"status": createStatus("FAILED"), "time": time.time() - start}
        ind += 1

    with open(dst_pkl, 'wb') as fw:
        pkl.dump(out, fw)
    print("Complete!")
    return outp_result

def writeRootPickles(src, dst_train, dst_valid, dest_training_pkl="dataTrainingRoot.pkl", dest_validation_pkl="dataValidationRoot.pkl", training_percentage=0.8, unique_mask_per_slice=False, skip_existing=True, verbose=False):
    olst = os.listdir(src)
    # Get only valid training data
    olst = [item for item in olst if item.endswith(".h5") or item.endswith(".im")]
    # only the last few are saved for validation, not exactly optimal
    d = {}
    d['train'] = createRootKSpace(src, dst_train, olst[:int(training_percentage * len(olst))], dest_training_pkl, unique_mask_per_slice=unique_mask_per_slice, skip_existing=skip_existing, verbose=verbose)
    print("=================================== STARTING VALIDATION CREATION ===================================")
    d['valid'] = createRootKSpace(src, dst_valid, olst[int(training_percentage * len(olst)) + 1:], dest_validation_pkl, unique_mask_per_slice=unique_mask_per_slice, skip_existing=skip_existing, verbose=verbose)
    # Returns a json of the states for all files that were either written or not, includes times for each conversion
    return d
