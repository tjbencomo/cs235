# Description: Load image data from pandas dataframe. Convert
# h5 data to numpy matrices and partition into train and test sets

import h5py as h5
import numpy as np
import pandas as pd

IMG_SHAPE = (256, 256)

def load_case(fp, return_mask = False):
    f = h5.File(fp, 'r')
    if f['data'][:, :, 0].shape != IMG_SHAPE:
        raise ValueError(f"{fp} does not match expected dimensions!")
    img = np.asarray(f['data'][:, :, 0])
    mask = np.asarray(f['data'][:, :, 1])
    if return_mask:
        return img, mask
    return img

def load_data(metadata, train_prop = .8):
    imgs = []
    for i, fp in enumerate(metadata['filepath']):
        img = load_case(fp)
        imgs.append(img)
    imgs = np.stack(imgs)
    labels = metadata['label'].tolist()
    labels = np.stack(labels)
    N_total = metadata.shape[0]
    N_train = int(N_total * train_prop)
    X_train, X_test = imgs[0:N_train], imgs[N_train:]
    Y_train, Y_test = labels[0:N_train], labels[N_train:]
    X_train = X_train.reshape((X_train.shape[0], 256, 256, 1))
    X_test = X_test.reshape((X_test.shape[0], 256, 256, 1))
    Y_train = Y_train.reshape((Y_train.shape[0], 1))
    Y_test = Y_test.reshape((Y_test.shape[0], 1))
    return (X_train, Y_train), (X_test, Y_test)

