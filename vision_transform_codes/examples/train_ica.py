"""
Train an ICA dictionary.

This shows some viable parameter settings for David Field's natural images
dataset. So long as you use the 'standardize_data_range' preprocessing step,
these should also work well for other datasets too
(see utils/dataset_generation.py).
"""
import _set_the_path

import pickle
import numpy as np
import torch

from training.ica import train_dictionary
from utils.dataset_generation import create_patch_training_set
from utils import defaults

RUN_IDENTIFIER = 'test_ICA'
LOGS_STORED_HERE = defaults.logging_directory

BATCH_SIZE = 250
NUM_BATCHES = 4000  # 1 million patches total
PATCH_HEIGHT = 16
PATCH_WIDTH = 16

CODE_SIZE = PATCH_HEIGHT * PATCH_WIDTH
NUM_EPOCHS = 10
iters_per_epoch = NUM_BATCHES

LOAD_DATA_FROM_DISK = False
if not LOAD_DATA_FROM_DISK:
  SAVE_TRAINING_DATA_FOR_FASTER_RELAUNCH = True
  patch_dataset = create_patch_training_set(
      num_batches=NUM_BATCHES, batch_size=BATCH_SIZE,
      patch_dimensions=(PATCH_HEIGHT, PATCH_WIDTH), edge_buffer=5,
      dataset='Field_NW',
      order_of_preproc_ops=['standardize_data_range',
                            'whiten_center_surround', 'patch'])
  if SAVE_TRAINING_DATA_FOR_FASTER_RELAUNCH:
    pickle.dump(patch_dataset, open(defaults.dataset_directory /
      'Field_natural_images/training_patches_sdr_white.p', 'wb'))
else:
  patch_dataset = pickle.load(open(defaults.dataset_directory /
    'Field_natural_images/training_patches_sdr_white.p', 'rb'))

ICA_PARAMS = {
    'num_epochs': NUM_EPOCHS,
    'dictionary_update_algorithm': 'ica_natural_gradient',
    'dict_update_param_schedule': {
      0: {'stepsize': 0.1, 'num_iters': 1},
      4 * NUM_BATCHES: {'stepsize': 0.01, 'num_iters': 1},
      8 * NUM_BATCHES: {'stepsize': 0.005, 'num_iters': 1}},
    # write various tensorboard logs on the following schedule:
    'training_visualization_schedule': {
      0: None, 500: None, 1000: None, 2000: None,
      'reshaped_kernel_size': (PATCH_HEIGHT, PATCH_WIDTH)},
    # actually store all logs here:
    'logging_folder_fullpath': LOGS_STORED_HERE / RUN_IDENTIFIER,
    # checkpoint the dictionary on this interval
    'checkpoint_schedule': {iters_per_epoch: None,
                            (NUM_EPOCHS*iters_per_epoch)-1: None}}
ICA_PARAMS['training_visualization_schedule'].update(
    {NUM_BATCHES*x: None for x in range(1, NUM_EPOCHS)})


# Now initialize modela and begin training
torch_device = torch.device('cuda:1')
torch.cuda.set_device(1)
# otherwise can put on 'cuda:0' or 'cpu'

# send ALL image patches to the GPU
image_patches_gpu = torch.from_numpy(
    patch_dataset['batched_patches']).to(torch_device)

# create the dictionary Tensor on the GPU
Q, R = np.linalg.qr(np.random.standard_normal((CODE_SIZE,
                                               PATCH_HEIGHT*PATCH_WIDTH)))
ica_dictionary = torch.from_numpy(Q.astype('float32')).to(torch_device)

print("Here we go!")
train_dictionary(image_patches_gpu, ica_dictionary, ICA_PARAMS)
# ^point a tensorboard session at 'logging_folder_fullpath' to see progress
