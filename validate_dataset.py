## Validator for datasets
## Author: Adam Noworolski (Sc2ad)
import pickle
import numpy as np
import os
import h5py

# Validation Errors are AssertionErrors

def validate_dataset(pickle_load, config):
    # For each set of files (input, output)
    assert type(pickle_load) == list, "pickle_load must be a list, not: " + str(type(pickle_load))
    for fileList in pickle_load:
        assert type(fileList) == list and len(fileList) == 2, "Each item in the pickle must be a list of size 2, not: " + str(type(fileList))
        # Open each file and confirm it matches the provided config
        check_files(file, config)

def assert_same(expected, test):
    assert expected == test, "Expected: " + str(expected) + " but got: " + str(test)

def check_files(fileArray, config):
    """
    Opens the given file, confirms that everything inside it is the same as defined by the config.
    File: a string that is the file to open and check
    Config: a dictionary that contains im_dims, num_classes, num_channels, etc.
    """
    assert type(config['idx_classes']) == list and len(config['idx_classes']) == config['num_classes'], "idx_classes must be a list of same size as num_classes"
    input_file = fileArray[0]
    output_file = fileArray[1]
    in_data = h5_check(input_file, config, dtype=np.float32)
    out_data = h5_check(output_file, config, dtype=np.uint8, crop=False)
    if out_data.ndim == 3:
        out_data = out_data[:, :, :, np.newaxis]
    out_data = out_data[:, :, :, config['idx_classes']]
    out_data = crop_data(out_data, config['crop'])

    #HEH!
    in_sanity = np.concatenate(([config['batch_size']],config['im_dims'],[config['num_channels']]))
    assert_same(in_sanity.shape, in_data.shape)

    out_sanity = np.concatenate(([config['batch_size']],config['im_dims'],[config['num_classes']]))
    assert_same(out_sanity.shape, out_data.shape)

def h5_check(file, config, dtype, crop=True):
    # Read the file as an h5py file
    with h5py.File(file, 'r') as f:
        assert "data" in f.keys(), "The h5py file must contain the 'data' key!"
        # TODO ADD DTYPE DETECTION
        data = np.array(f['data'], dtype=dtype)
        assert data.ndim * 2 == len(config['crop']), "The length of the crop must be twice the size of the length of the shape of the image, not: " + str(len(config['crop']))
        for i in range(0, len(config['crop']), 2):
            assert data.shape[i//2] > config['crop'][i] + config['crop'][i + 1], "Crop cannot be larger than the original image!"
        if crop:
            data = crop_data(data, config['crop'])
        assert data.shape[0] == config['im_dims'][0], "image x-dimensions must match!"
        assert data.shape[1] == config['im_dims'][1], "image y-dimensions must match!"
        if data.ndim == 3:
            # Will throw an index out of bounds if config['im_dims'] does not have index 2, but that's okay.
            assert data.shape[2] == config['im_dims'][2], "image z-dimensions must match!"
    return data

def crop_data(data, crop):
    if crop[1] == 0:
        data = data[crop[0]:,...]
    else:
        data = data[crop[0]:-crop[1],...]
    if crop[3] == 0:
        data = data[:,crop[2]:,...]
    else:
        data = data[:,crop[2]:-crop[3],...]
    if crop[5] == 0:
        data = data[:,:,crop[4]:,...]
    else:
        data = data[:,:,crop[4]:-crop[5],...]
    return data