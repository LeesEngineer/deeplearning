#Initializing parameters, Calculating the cost function and its gradient, Using an optimization algorithm (gradient descent)

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# (m_train, num_px, num_px, 3)
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# _orig need to be preprocessed
# A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b*c*d, a) is to use:
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T

# One common preprocessing step in machine learning is to center and standardize your dataset
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255





 
































