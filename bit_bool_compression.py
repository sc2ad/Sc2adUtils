from bitstring import BitArray
import argparse
import random
import numpy as np
import pickle
import struct

def readBool8(reader):
    b = reader.read(1)
    if b == b'':
        return 0
    return parseByte(b)

def readHeader(reader):
    dims = []
    v = struct.unpack('I', reader.read(4))[0]
    while v != 0:
        dims.append(v)
        v = struct.unpack('I', reader.read(4))[0]
    return dims

def parseByte(b):
    if type(b) != bytes:
        b = bytes([b])
    o = BitArray(b).bin
    bools = []
    for i in range(len(o)):
        bools.append(o[i] == '1')
    return bools

def getbytes(bits):
    if type(bits) == list:
        bits = iter(bits)
    done = False
    while not done:
        byte = 0
        for _ in range(0, 8):
            try:
                bit = next(bits)
            except StopIteration:
                bit = 0
                done = True
            byte = (byte << 1) | bit
        if not done:
            yield byte

def writeHeader(writer, dims):
    for item in dims:
        writer.write(struct.pack('I', item))
    writer.write(struct.pack('I', 0))

def writeBools(writer, bools):
    o = [1 if item else 0 for item in bools]
    while len(o) % 8 != 0:
        o.append(0)
    for b in getbytes(o):
        writer.write(bytes([b]))

def from_numpy(numpy_arr):
    if type(numpy_arr) == np.uint8:
        return [np.uint8(item) for item in parseByte(numpy_arr)]
    assert numpy_arr.dtype == np.uint8, "Numpy array must be a uint8 array!"
    if numpy_arr.ndim == 1:
        # Because there is more than one dimension to this array, we must travel "down" to the lowest dimension
        # before we can check the header length of the array

        assert len(numpy_arr) > 4, "Numpy array cannot be empty! (must contain 4 byte header)"
        # First 4 bytes are used for proper size of the array
        length = struct.unpack("I", bytes(numpy_arr[:4]))[0]
        l = [from_numpy(item) for item in numpy_arr[4:]]
        # Take the very last item of the array and shorten it to the proper length.
        l[(length - 1)//8] = l[(length - 1)//8][:length % 8 if length % 8 != 0 else 8]
        l = sum(l, [])
        new = np.array(l).reshape((length,))
    else:
        l = [from_numpy(item) for item in numpy_arr]
        new = np.array(l)
    return new

def to_numpy(bools):
    if (type(bools) == list or type(bools) == np.ndarray) and (type(bools[0]) == bool or type(bools[0]) == np.bool_ or type(bools[0]) == np.uint8):
        o = [1 if item else 0 for item in bools]
        while len(o) % 8 != 0:
            o.append(0)
        length_bytes = struct.pack("I", len(bools))
        return np.array([b for b in length_bytes] + list(getbytes(o)), dtype=np.uint8)
    assert type(bools) == list or type(bools) == np.ndarray, "Bools must be a list of bools, not: " + str(type(bools))
    new = np.array([list(to_numpy(b)) for b in bools], dtype=np.uint8)
    return new

def roundTripTest(file, boolMax):
    with open(file, 'wb') as f:
        boolCount = random.randint(1, boolMax)
        writeHeader(f, [boolCount])
        bs = []
        for i in range(boolCount):
            bs.append(random.randint(0, 1) == 1)
        writeBools(f, bs)
    with open(file, 'rb') as f:
        dims = readHeader(f)
        bools = readBool8(f)
        b = []
        while bools:
            b += bools
            bools = readBool8(f)
        b = b[:dims[0]]
    assert dims[0] == boolCount, "Dimensions should match!"
    assert b[:len(bs)] == bs, "Bool arrays should match but don't!"

def roundTripTestNumpy(npFile, bools):
    np1 = to_numpy(bools)
    with open(npFile, 'wb') as f:
        pickle.dump(np1, f)
    bs1 = from_numpy(np1)
    temp = to_numpy(bs1)
    assert temp.all() == np1.all(), "'From' conversion should not fail!"
    with open(npFile, 'rb') as f:
        np2 = pickle.load(f)
    bs2 = from_numpy(np2)
    temp = to_numpy(bs2)
    assert temp.all() == np2.all(), "'From' conversion should not fail!"
    assert np1.all() == np2.all(), "Numpy arrays should match but don't!"

def roundTripTestNumpyMulti(npFile, boolMax, dimCount=1):
    bools = []
    for j in range(dimCount):
        boolCount = random.randint(1, boolMax)
        bs = []
        for i in range(boolCount):
            bs.append(random.randint(0, 1) == 1)
        bools.append(bs)
    print(bools)
    roundTripTestNumpy(npFile, bools)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Writes/Reads bools in an 8 bit compressed state from the provided files.")
    parser.add_argument("file")
    parser.add_argument("--read", default=False)
    parser.add_argument("--boolMax", default=20, type=int)
    parser.add_argument("--dims", default=2, type=int)

    args = parser.parse_args()

    roundTripTest(args.file, args.boolMax)
    roundTripTestNumpyMulti(args.file + ".pkl", args.boolMax, dimCount=args.dims)

    if args.read:
        b = []
        with open(args.file, 'rb') as f:
            # First read the dims
            dims = readHeader(f)
            bools = readBool8(f)
            while bools:
                b += bools
                bools = readBool8(f)
            b = b[:dims[0]]
        print(b)
    else:
        with open(args.file, 'wb') as f:
            boolCount = random.randint(1, args.boolMax)
            writeHeader(f, [boolCount])
            bs = []
            for i in range(boolCount):
                bs.append(random.randint(0, 1) == 1)
            writeBools(f, bs)
        print(bs)
        