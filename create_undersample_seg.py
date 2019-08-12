import os
import pickle as pkl
import sys

#f = '/data/knee_mri2/DioscoriDESS_temp/Dioscorides/CCsplits/vanillaCC_cv_split3_val.pickle'
f = '/data/knee_mri5/Adam/fastMRI_Data/Validation/validRoot_2_complex.pkl'

# search_dirs = ["/data/knee_mri5/Adam/fastMRI_Data/Validation/original", "/data/knee_mri5/Adam/fastMRI_Data/Training/original"]

def create(dst, accels=[2,3,5,10,12], srcV='/data/knee_mri5/Adam/fastMRI_Data/Validation/', srcT='/data/knee_mri5/Adam/fastMRI_Data/Training/'):

	with open(f, 'rb') as q:
		arr = pkl.load(q)
	for a in accels:
		search_dirs = [srcV + 'undersampled_' + str(a) + '_complex', srcT + 'undersampled_' + str(a) + '_complex']
		n_arr = []

		for item in arr:
			for p in search_dirs:
				path = os.path.join(p, os.path.split(item[0])[-1].replace(".im", "_undersampled.im"))
				#print(path)
				if os.path.exists(path):
				    n_arr.append([path, item[1]])
				    print("Added item with path: " + path)
		with open(dst + '_' + str(a) + ".pkl", 'wb') as d:
			print("Created file: " + dst + "_" + str(a) + ".pkl")
			pkl.dump(n_arr, d)

def replace_images(src, dst):
    new_arr = []
    pickles = ['/data/knee_mri2/DioscoriDESS_temp/Dioscorides/CCsplits/vanillaCC_cv_split3_val.pickle', '/data/knee_mri2/DioscoriDESS_temp/Dioscorides/CCsplits/vanillaCC_cv_split3_train.pickle']
    for item in os.listdir(src):
        for pick in pickles:
            found = False
            with open(pick, 'rb') as q:
                arr = pkl.load(q)
            for s in arr:
                if item.replace("_undersampled", "") in s[0]:
                    new_arr.append([os.path.abspath(os.path.join(src, item)), s[1]])
                    print("Added new item with image path: " + os.path.join(src, item))
                    found = True
                    break
            if found:
                break
    with open(os.path.join(dst, "reconstructed.pkl"), 'wb') as d:
        print("Created file: " + os.path.join(dst, "reconstructed.pkl"))
        pkl.dump(new_arr, d)

#if len(sys.argv) > 1:
#    create(sys.argv[1])
