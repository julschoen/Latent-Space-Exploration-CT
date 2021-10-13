import numpy as np
from pydicom import dcmread
import os
import cv2 as cv
import shutil

# Define Paths
# Path to original LIDC-IDRI Dataset
data_location = 'path'
# Path where Train Test split is supposed to be saved
save_location = 'path/'

# Path for NPZ-Sets
train = 'path'
test = 'path'

# Path for final preprocessed dataset
train_save = 'path'
test_save = 'path'


def train_test_split(data_path, save, train_names, test_names):
    """
    splits into train and test data folders
    """
    for dirs in os.listdir(data_path):
        if not os.path.isdir(os.path.join(data_path, dirs)) \
                or not dirs.startswith('LIDC-IDRI-'):
            continue

        if dirs in test_names:
            save_path = os.path.join(save, 'Test')
        else:
            save_path = os.path.join(save, 'Train')
        shutil.move(os.path.join(data_path, dirs), save_path)


def dcm2png(file, save_location, cntr):
    ds = dcmread(file)

    shape = ds.pixel_array.shape

    intercept = float(ds[0x28, 0x1052].value)
    slope = float(ds[0x28, 0x1053].value)
    image_2d = ds.pixel_array.astype('float32')

    image_2d = (image_2d * slope) + intercept
    image_2d = np.clip(image_2d, -1000, 2000)

    path = os.path.join(save_location, f'{cntr}.npz')
    np.savez_compressed(path, image_2d)


def get_subdirs(d):
    return [o for o in os.listdir(d)
            if os.path.isdir(os.path.join(d, o))]


def make_set(data_path, new_data, dir_name=None, cntr=0):
    """
    Takes LIDR dataset as downloaded from TCIA and creates npz files in patient dirs
    NPZs are already clipped and rescaled using header information
    """
    for dirs in os.listdir(data_path):
        if not os.path.isdir(os.path.join(data_path, dirs)):
            continue
        if dir_name == None:
            dir_name = dirs[-4:]

        subdir = get_subdirs(os.path.join(data_path, dirs) + '/')

        if len(subdir) == 0:
            files = [f for f in os.listdir(data_path + dirs) if os.path.isfile(os.path.join(data_path + dirs, f))]
            for filename in files:
                if filename.endswith('.dcm'):
                    dcm2png(os.path.join(data_path, dirs, filename), new_data, cntr)
                    cntr += 1
        else:
            for d in subdir:
                cntr = make_set(os.path.join(data_path, dirs, '/'), new_data, dir_name, cntr)
        dir_name = None
    return cntr


def save_patient_nums(directory, location):
    """
    Saves the filenames of patients for a given dataset
    """
    names = []
    for name in os.listdir(directory):
        if name.startswith('LIDC-IDRI'):
            names.append(name)
    names = np.array(names)
    np.savetxt(os.path.join(location, 'patien_nums.csv'), names, fmt='%s')


def preprocess(data_path, size=128, normalize=True):
    """
    Resizes, and normalizes data using simple min-max-scaling (-1,1)
    """
    print('Get a coffee this will take a while ...')
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    for filename in files:
        if filename.endswith('.npz'):
            path = os.path.join(data_path, filename)
            x = np.load(path)['arr_0']

            if size is not None:
                x = cv.resize(x, (size, size), interpolation=cv.INTER_LANCZOS4)
            if normalize:
                # Simple min max scaling:
                x = (x + 1000) / 3000
                x = x * 2 - 1
                x = np.clip(x, -1, 1)
            np.savez_compressed(path, x.astype('float32'))


# Split into train and test set
train_names = np.loadtxt(os.path.join(save_location, 'patien_nums_train.csv'), dtype=str)
test_names = np.loadtxt(os.path.join(save_location, 'patien_nums_test.csv'), dtype=str)
train_test_split(data_location, save_location, train_names, test_names)

# Scale turn dicom to npz
make_set(test, test_save)
make_set(train, train_save)

# Resize and normalize
preprocess(train_save)
preprocess(test_save)
