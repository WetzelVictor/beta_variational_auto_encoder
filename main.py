#-*-encoding:UTF-8-*-
"""
Beta Variational Auto-Encoder
"""

#%% Librairies
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision

from framework import modVAE
from framework.utils import to_var, zdim_analysis

from toyDataset import dataset as dts
import matplotlib.pyplot as plt
from numpy.random import randint
import numpy as np

import librosa

import os.path as pa
import pickle

#%% PARAMETERS
# Parameters, dataset
N_FFT = 100
BATCH_SIZE = 50
#LEN_EXAMPLES = 38400
LEN_EXAMPLES = 2000
# Net parameters
Z_DIM, H_DIM = 20, 400
FS = 8000

#%% Importing DATASET

# Creating dataset
DATASET_FILEPATH = 'data/datasets/DATASET_test.obj'

# If there is no archive of the dataset, it needs to be rendered
if not pa.isfile(DATASET_FILEPATH):
    print 'Generating Dataset \n\n'
    DATASET = dts.toyDataset(length_sample=LEN_EXAMPLES,
                            n_bins=N_FFT,
                            Fe_Hz=FS,
                            data='cqt')
    obj = DATASET
    file_obj = open(DATASET_FILEPATH, 'w')
    pickle.dump(obj, file_obj)
    print 'File is {0}'.format(DATASET_FILEPATH)
else:
    # Otherwise, load the pickled archive
    print 'Importing dataset at location' + DATASET_FILEPATH
    DATASET = pickle.load(open(DATASET_FILEPATH,'rb'))

IMG_LENGTH = np.shape(DATASET.__getitem__(9)[0])[1]
for i in xrange(45, 75):
    if (IMG_LENGTH % i) == 0:
        BATCH_SIZE = i
        print('Mini_batch size is %d'%(BATCH_SIZE))
        break

DATA_LOADER = torch.utils.data.DataLoader(dataset=DATASET,
                                          batch_size =BATCH_SIZE,
                                          shuffle=True)

#%% Saving original image
FIXED_INDEX = randint(BATCH_SIZE)

# Saving an item from the dataset to debug
DATA_ITER = iter(DATA_LOADER)
FIXED_X,_ = next(DATA_ITER)
FIXED_X = torch.Tensor(FIXED_X.float()).view(FIXED_X.size(0), -1).squeeze()
HEIGHT,WIDTH = FIXED_X.size()

#%% SAVING fixed x as an image
FIXED_X = to_var(FIXED_X)
reconst_images = FIXED_X.view(BATCH_SIZE,1,N_FFT,-1)
torchvision.utils.save_image(reconst_images.data.cpu(),'./data/CQT/original_images.png')


#%% CREATING THE Beta-VAE
betaVAE = modVAE.VAE(image_size=WIDTH, z_dim=Z_DIM, h_dim=H_DIM)

# BETA: Regularisation factor
# 0: Maximum Likelihood
# 1: Bayes solution
BETA = 4


# GPU computing if available
if torch.cuda.is_available():
    betaVAE.cuda()
    print('GPU acceleration enabled')

# OPTIMIZER
OPTIMIZER = torch.optim.Adam(betaVAE.parameters(), lr=0.001)

ITER_PER_EPOCH = len(DATASET)/BATCH_SIZE
NB_EPOCH = 10;


#%%
""" TRAINING """
for epoch in range(NB_EPOCH):    
    # Epoch
    print ' '
    print '\t \t  /=======================================\\'
    print '\t \t  | - | - | - | EPOCH [%d/%d] | - | - | - | '%(epoch+1, NB_EPOCH)
    print '\t \t  \\=======================================/'
    print ' '
    for i, (images, params) in enumerate(DATA_LOADER):

        # Formatting
        images = to_var(torch.Tensor(images.float()).squeeze())
        out, mu, log_var = betaVAE(images)

        # Compute reconstruction loss and KL-divergence
        reconst_loss = F.binary_cross_entropy(out, images, size_average=True)

        kl_divergence = torch.sum(0.5 * (mu**2
                                         + torch.exp(log_var)
                                         - log_var -1))

        # Backprop + Optimize
        total_loss = reconst_loss + BETA*kl_divergence
        OPTIMIZER.zero_grad()
        total_loss.backward()
        OPTIMIZER.step()

        # PRINT
        # Prints stats at each epoch
        if i%10 == 0:
            print ("Step [%d/%d] \t Total Loss: %.2f \t Reconst Loss: %.2f \t KL Div: %.3f"
                   %(i,
                     ITER_PER_EPOCH,
                     total_loss.data[0],
                     reconst_loss.data[0],
                     kl_divergence.data[0])
                  )

    # Save the reconstructed images
    reconst_images, _, _ = betaVAE(FIXED_X)
    reconst_images = reconst_images.view(BATCH_SIZE, 1, N_FFT, -1)
    torchvision.utils.save_image(reconst_images.data.cpu(),
                                 './data/CQT/reconst_images_%d.png' %(epoch+1))

#%% SAMPLING FROM LATENT SPACE
#FIXED_Z = zdim_analysis(BATCH_SIZE, Z_DIM, 50, -20, 20)
#FIXED_Z = to_var(torch.Tensor(FIXED_Z.contiguous()))
#
## Sampling from model, reconstructing from spectrogram
#sampled_image = betaVAE.sample(FIXED_Z)
#reconst_images = sampled_image.view(BATCH_SIZE, 1, N_FFT, -1)
#torchvision.utils.save_image(reconst_images.data.cpu(), './data/CQT/sampled_images.png')
#
##%%
#sampled_image_numpy = sampled_image.data.numpy()
#sampled_image_numpy =sampled_image_numpy[1,:].reshape(N_FFT/2+1,-1)
#
#reconst_sound = DATASET.audio_engine.griffinlim(sampled_image_numpy, N_iter=500)
#output_name = 'sampled_sound.wav'
#librosa.output.write_wav('data/SOUND/sampled_sound.wav',reconst_sound,DATASET.Fs)

