import matplotlib.pyplot as plt
import numpy as np
import h5py
from copy import copy

def load_seg_masks(seg, key_slice=90):
    s = seg[:, :, key_slice]
    s = np.reshape(s, (344,344,5))
    s[s > 0.5] = 1
    s[s <= 0.5] = 0
    return s

def plot(src, seg, key_slice=90):
    fig, ax = plt.subplots(constrained_layout=True)
    plt.imshow(np.reshape(src[:, :, key_slice], (src.shape[0], src.shape[1])), cmap=plt.cm.gray, alpha=1.0)
    masks = load_seg_masks(seg, key_slice=key_slice-10)

    colmax = np.argmax(masks, axis=2)
    colmax = np.ma.masked_where(colmax == 4, colmax)
    #c = ax.contourf(mask, colors=colors[i])
    plt.imshow(colmax, cmap='rainbow')
    plt.show()

def load(src, seg, key_slice=110):
    if src.endswith(".im"):
        with h5py.File(src, 'r') as f:
            data = np.array(np.abs(f['data']))
        with np.load(seg) as d:
            segmented = d['pred']
            print(segmented.shape)
        plot(data, segmented, key_slice=key_slice)
    else:
        print("Can't load non-h5py files!")
# Reconstructed (12)
#f1 = '/data/knee_mri5/Adam/fastMRI_Data/Prediction/Reconstructed_12/H5Images/pat_062_V01_undersampled.im'
# Original
f1 = '/data/knee_mri4/DESS_data/vanillaCC_fixed/pat_062_V01.im'
# Undersampled (12)
#f2 = '/data/knee_mri5/Adam/fastMRI_Data/Prediction/Undersampled_12/0808113955_adam_smallNet_under_12/pat_062_V01_undersampled.npz'
# Original
#f2 = '/data/knee_mri5/Adam/fastMRI_Data/Prediction/Original/0807190342_adam_smallNet_original/pat_062_V01.npz'
# Reconstructed (12)
f2 = '/data/knee_mri5/Adam/fastMRI_Data/Prediction/Reconstructed_12/0808144713_adam_smallNet_rec_12/pat_062_V01_undersampled.npz'
load(f1, f2)
