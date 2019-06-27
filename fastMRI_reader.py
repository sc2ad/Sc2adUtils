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
    return np.array(np.abs(fft2c(kspace)), dtype=np.float32)

def convert_to_kspace(image):
    return np.array(ifft2c(image))

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

def createRootKSpace(src, dst_orig, dst_under, lst, dst_pkl, replicate_orig=False, unique_mask_per_slice=False, skip_existing=True, verbose=False):
    rate = 10
    accelF = 12
    out = []
    outp_result = {}
    ind = 0
    for f in lst:
        start = time.time()
        original = os.path.abspath(os.path.join(src, f))
        print("Starting file: " + original + "(" + str(ind) + "/" + str(len(lst)) + ")")
        path = os.path.abspath(os.path.join(dst_under, f.replace(".im", "_undersampled.im").replace(".h5", "_undersampled.im")))
        label = original
        if replicate_orig:
            label = os.path.abspath(os.path.join(dst_orig, f.replace(".im", "_original.im").replace(".h5", "_original.im")))
        if skip_existing and os.path.exists(path) and os.path.exists(label):
            print("Skipping file because it already exists!")
            out.append([path, label])
            outp_result[original] = {"status": createStatus("SUCCESS_SKIP"), "time": time.time() - start}
            ind += 1
            continue
        try:
            # Input is a new file that needs to be generated
            # Needs to be in the image space, convert to kspace, undersample,
            # then convert from kspace to image space
            d = read_h5_unsafe(original)
            # kspace = readKSpace(d)
            # image = convert_to_image(kspace)
            image = np.array(readImage(d))
            new_image = image.copy()
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
                    kspace = convert_to_kspace(image[:, :, i])
                    image[:, :, i] = convert_to_image(kspace * mask)
                if replicate_orig:
                    kspace = convert_to_kspace(new_image[:, :, i])
                    new_image[:, :, i] = convert_to_image(kspace)
            with h5py.File(path, 'w') as fw:
                # fw.create_dataset("data", image.shape, dtype='f4')
                fw['data'] = image
            if replicate_orig:
                with h5py.File(label, 'w') as fw:
                    fw['data'] = new_image

            d['_file'].close()
            out.append([path, label])
            delta = time.time() - start
            print("Completed file: " + original + " in: " + str(delta) + " seconds!")
            outp_result[original] = {"status": createStatus("FULL_SUCCESS"), "time": delta}
        except KeyboardInterrupt:
            print("Halting!")
            raise KeyboardInterrupt
        except:
            print("Failed to create file at path: " + path + " from original: " + original + " with label: " + label + "!")
            outp_result[original] = {"status": createStatus("FAILED"), "time": time.time() - start}
        ind += 1

    with open(dst_pkl, 'wb') as fw:
        pkl.dump(out, fw)
    print("Complete!")
    return outp_result

def writeRootPickles(src, dst_train_orig, dst_train_under, dst_valid_orig, dst_valid_under, dest_training_pkl="dataTrainingRoot.pkl", dest_validation_pkl="dataValidationRoot.pkl", training_percentage=0.8, replicate_orig=False, unique_mask_per_slice=False, skip_existing=True, verbose=False):
    olst = os.listdir(src)
    # Get only valid training data
    olst = [item for item in olst if item.endswith(".h5") or item.endswith(".im")]
    # only the last few are saved for validation, not exactly optimal
    d = {}
    d['train'] = createRootKSpace(src, dst_train_orig, dst_train_under, olst[:int(training_percentage * len(olst))], dest_training_pkl, replicate_orig=replicate_orig, unique_mask_per_slice=unique_mask_per_slice, skip_existing=skip_existing, verbose=verbose)
    print("=================================== STARTING VALIDATION CREATION ===================================")
    d['valid'] = createRootKSpace(src, dst_valid_orig, dst_valid_under, olst[int(training_percentage * len(olst)) + 1:], dest_validation_pkl, replicate_orig=replicate_orig, unique_mask_per_slice=unique_mask_per_slice, skip_existing=skip_existing, verbose=verbose)
    # Returns a json of the states for all files that were either written or not, includes times for each conversion
    return d

# a = r.writeRootPickles("/data/knee_mri4/DESS_data/vanillaCC_fixed", "/data/knee_mri5/Adam/fastMRI_Data/Training/original", "/data/knee_mri5/Adam/fastMRI_Data/Training/undersampled_12", "/data/knee_mri5/Adam/fastMRI_Data/Validation/original", "/data/knee_mri5/Adam/fastMRI_Data/Validation/undersampled_12")
