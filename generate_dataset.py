#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" generate_dataset.py
This script is designed to generate the dataset, and then save it in a pickle archive.
The dataset is an object. See 'toyDataset/dataset.py' for methods and data
"""
#%% IMPORTS STATEMENTS
from framework import modAttentiondef
from framework.utils import to_var, zdim_analysis

from toyDataset import dataset as dts
import matplotlib.pyplot as plt
from numpy.random import randint
import numpy as np

import librosa

import os.path as pa
import pickle


# %% PARAMETERS
# Parameters, dataset
N_FFT = 100
BATCH_SIZE = 50
#LEN_EXAMPLES = 38400
LEN_EXAMPLES = 2000
# Net parameters
Z_DIM, H_DIM = 20, 400
FS = 8000

# %% Importing DATASET

# Creating dataset
DATASET_FILEPATH = 'data/datasets/DATASET.obj'

# If there is no archive of the dataset, it needs to be rendered
if not pa.isfile(DATASET_FILEPATH):
    print 'Generating Dataset \n\n'
    DATASET = dts.toyDataset(length_sample=LEN_EXAMPLES,
                             n_bins=N_FFT,
                             Fe_Hz=FS,
                             data='spectro')
    obj = DATASET
    file_obj = open(DATASET_FILEPATH, 'w')
    pickle.dump(obj, file_obj)
    print 'File is {0}'.format(DATASET_FILEPATH)
else:
    # Otherwise, load the pickled archive
    # print 'Importing dataset at location {}'%(DATASET_FILEPATH)
    DATASET = pickle.load(open(DATASET_FILEPATH, 'rb'))

