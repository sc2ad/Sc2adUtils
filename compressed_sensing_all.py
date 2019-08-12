import h5py
import numpy as np
import os
import pickle as pkl
import compressed_sensing_recon as cr
import fastMRI_reader as r
import time

def recon_all(src, dst, accel=2):
    times = []
    if not os.path.exists(dst):
        os.mkdir(dst)
    for item in os.listdir(src):
        original_name = os.path.abspath(os.path.join(src, item))
        d = r.read_h5_unsafe(original_name)
        # kspace = readKSpace(d)
        # image = convert_to_image(kspace)
        image = np.array(r.readImage(d)).astype(dtype=np.complex128)
        new_image = image.copy()
        start = time.time()
        for i in range(image.shape[-1]):
            new_image[:, :, i] = cr.image_undersampled_recon(image[:, :, i], accel_factor=accel)
        # recon = cr.image_undersampled_recon(new_image, accel_factor=accel)
        with h5py.File(os.path.abspath(os.path.join(dst, item)), 'w') as fw:
            # fw.create_dataset("data", image.shape, dtype='f4')
            fw['data'] = new_image
        d['_file'].close()
        delta = time.time() - start
        print("Completed file: " + original_name + " in: " + str(delta) + " seconds!")
        times.append(delta)
    print("Complete!")
    print("Average time: " + str(sum(times) / float(len(times))))

def recon_range(src, dst, accel=[2,3,5,10,12]):
    accel = accel[::-1]
    for a in accel:
        print("Starting accel factor: " + str(a))
        path = os.path.join(dst, "accel_" + str(a))
        if not os.path.exists(path):
            os.mkdir(path)
        recon_all(src, path, accel=a)
        print("Completed accel factor: " + str(a))
    print("Completed all accel factors!")

def convert_npz_im(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    for item in os.listdir(src):
        with np.load(os.path.abspath(os.path.join(src, item))) as data:
            with h5py.File(os.path.abspath(os.path.join(dst, item)).replace(".npz", ".im"), 'w') as fw:
                #print(data['pred'].shape)
                fw['data'] = data['pred']
        print("Created file: " + os.path.abspath(os.path.join(dst, item)).replace(".npz", ".im"))
    for item in os.listdir(dst):
        with h5py.File(os.path.join(dst, item)) as fr:
            print(np.array(fr['data']).shape)
    print("Complete!")

def convert_npz_im_many(srcs=[], dsts=[]):
    for i in range(len(srcs)):
        convert_npz_im(srcs[i], dsts[i])
