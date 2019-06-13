## Validator for datasets
## Author: Adam Noworolski (Sc2ad)
import pickle
import numpy as np
import os
import h5py
import argparse

ASSERTIONS = False
# Validation Errors are AssertionErrors

def validate_dataset(pickle_load, config, verbose=False, printMod=20):
    # For each set of files (input, output)
    assert_true(type(pickle_load) == list, "pickle_load must be a list, not: " + str(type(pickle_load)))
    m = np.Infinity
    M = -np.Infinity
    for i in range(len(pickle_load)):
        # Open each file and confirm it matches the provided config
        new_min, new_max = check_files(pickle_load[i], config)
        m = min(new_min, m)
        M = max(new_max, M)
        if i % printMod == 0 and verbose:
            print("Completed validation for file: " + str(i) + " of: " + str(len(pickle_load)) + " overall min: " + str(m) + " overall max: " + str(M))

def assert_true(truth, message=""):
    # TODO Replace with custom assertions
    if ASSERTIONS:
        assert truth, message
    else:
        if not truth:
            print("AssertionError: " + message)

def assert_same(expected, test):
    assert_true(expected == test, "Expected: " + str(expected) + " but got: " + str(test))

def check_files(fileArray, config):
    """
    Opens the given fileArray, confirms that everything inside it is the same as defined by the config.
    FileArray: a list of strings that is the file to open and check
    Config: a dictionary that contains im_dims, num_classes, num_channels, etc.
    """
    assert_true(type(fileArray) == list and len(fileArray) == 2, "Each item in the pickle must be a list of size 2, not: " + str(type(fileArray)))
    assert_true(type(config['idx_classes']) == list, "idx_classes must be a list")
    input_file = fileArray[0]
    output_file = fileArray[1]
    in_data = h5_check(input_file, config, dtype=np.float32)
    out_data = h5_check(output_file, config, dtype=np.uint8, crop=False)
    uniques = np.unique(out_data)
    assert_true(len(uniques) == 2 and 0 in uniques and 1 in uniques, "The image must ONLY contain a binary mask! (0 and 1 ONLY). Instead, it contains the following types of values: " + str(uniques))
    assert_true(len(config['idx_classes']) + 1 <= out_data.shape[len(out_data.shape) - 1], "idx_classes must be <= the channels of the image: " + str(len(config['idx_classes']) + 1) + "<=" + str(out_data.shape[len(out_data.shape) - 1]))
    seg = np.zeros((out_data.shape[0], out_data.shape[1], out_data.shape[2], len(config['idx_classes']))).astype('uint8')

    seg[:,:,:,:-1] = out_data[:,:,:,config['idx_classes'][:-1]]
    seg[...,-1] = 1-np.max(seg,axis=3)
    out_data = crop_data(seg, config['crop'])

    #HEH!
    # TODO CONFIRM INPUT NUM CHANNELS, OUTPUT NUM CLASSES ARE LEGAL
    # in_sanity = np.concatenate(([config['batch_size']],config['im_dims'],[config['num_channels']]))
    # Possibly only works because num_channels = 1, batch_size = 1?
    in_sanity = tuple(config['im_dims'])
    assert_same(in_sanity, in_data.shape)

    # out_sanity = np.concatenate(([config['batch_size']],config['im_dims'],[config['num_classes']]))
    # Possibly only works because batch_size = 1?
    out_sanity = tuple(np.concatenate((config['im_dims'], [config['num_classes']])).tolist())
    assert_same(out_sanity, out_data.shape)
    return np.min(in_data), np.max(in_data)

def h5_check(file, config, dtype, crop=True):
    # Read the file as an h5py file
    with h5py.File(file, 'r') as f:
        assert_true("data" in f.keys(), "The h5py file must contain the 'data' key!")
        # Added DTYPE detection
        assert_true(f['data'].dtype == dtype, "Type of h5py must be: " + str(dtype) + " but was: " + str(f['data'].dtype))
        data = np.array(f['data'], dtype=dtype)
        assert_true(6 == len(config['crop']), "The length of the crop must be 6, not: " + str(len(config['crop'])))
        for i in range(0, len(config['crop']), 2):
            assert_true(data.shape[i//2] > config['crop'][i] + config['crop'][i + 1], "Crop cannot be larger than the original image!")
        if crop:
            data = crop_data(data, config['crop'])
        assert_true(data.shape[0] == config['im_dims'][0], "image x-dimensions must match!")
        assert_true(data.shape[1] == config['im_dims'][1], "image y-dimensions must match!")
        if data.ndim == 3:
            # Will throw an index out of bounds if config['im_dims'] does not have index 2, but that's okay.
            assert_true(data.shape[2] == config['im_dims'][2], "image z-dimensions must match!")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attempts to validate the given pickle file and config file")
    parser.add_argument("config")
    parser.add_argument("--verbose", default=False, type=bool)
    parser.add_argument("--printMod", default=20, type=int)
    args = parser.parse_args()

    from validate_config import *
    config = loadYaml(args.config)
    with open(config['data_train']['data_root'], 'rb') as f:
        validate_dataset(pickle.load(f), config['data_train'], verbose=args.verbose, printMod=args.printMod)