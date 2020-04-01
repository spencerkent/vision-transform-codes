"""
Train a sparse coding dictionary.

This shows some viable parameter settings for David Field's natural images
dataset. So long as you use the 'standardize_data_range' preprocessing step,
these should also work well for other datasets too
(see utils/dataset_generation.py).
"""
import _set_the_path

import pickle
import torch

from training.sparse_coding import train_dictionary
from utils.dataset_generation import create_patch_training_set
from utils import defaults

RUN_IDENTIFIER = 'test_sparse_coding'
LOGS_STORED_HERE = defaults.logging_directory

BATCH_SIZE = 250
NUM_BATCHES = 4000  # 1 million patches total
PATCH_HEIGHT = 16
PATCH_WIDTH = 16

CODE_SIZE = 1 * PATCH_HEIGHT*PATCH_WIDTH  # critically sampled
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

SC_PARAMS = {
    'mode': 'fully-connected',
    'num_epochs': NUM_EPOCHS,
    'code_inference_algorithm': 'fista',
    'inference_param_schedule': {
      0: {'sparsity_weight': 0.02, 'num_iters': 25},
      2*iters_per_epoch: {'sparsity_weight': 0.02, 'num_iters': 50},
      5*iters_per_epoch: {'sparsity_weight': 0.02, 'num_iters': 100}},
    'dictionary_update_algorithm': 'sc_cheap_quadratic_descent',
    'dict_update_param_schedule': {
      0: {'stepsize': 0.1, 'num_iters': 1},
      5*iters_per_epoch: {'stepsize': 0.05, 'num_iters': 1}},
    # write various tensorboard logs on the following schedule:
    'training_visualization_schedule': {
      0: None, 500: None, 1000: None, 2000: None,
      'reshaped_kernel_size': (PATCH_HEIGHT, PATCH_WIDTH)},
    # actually store all logs here:
    'logging_folder_fullpath': LOGS_STORED_HERE / RUN_IDENTIFIER,
    # checkpoint the dictionary on this interval
    'checkpoint_schedule': {iters_per_epoch: None,
                            (NUM_EPOCHS*iters_per_epoch)-1: None}}
SC_PARAMS['training_visualization_schedule'].update(
    {iters_per_epoch*x: None for x in range(1, NUM_EPOCHS)})


# Now initialize modela and begin training
torch_device = torch.device('cuda:1')
torch.cuda.set_device(1)
# otherwise can put on 'cuda:0' or 'cpu'

# send ALL image patches to the GPU
image_patches_gpu = torch.from_numpy(
    patch_dataset['batched_patches']).to(torch_device)

# create the dictionary Tensor on the GPU
sparse_coding_dictionary = torch.randn((CODE_SIZE, PATCH_HEIGHT*PATCH_WIDTH),
                                       device=torch_device)
# start out the dictionaries with norm 1
sparse_coding_dictionary.div_(
    sparse_coding_dictionary.norm(p=2, dim=1)[:, None])

print("Here we go!")
train_dictionary(image_patches_gpu, sparse_coding_dictionary, SC_PARAMS)
# ^point a tensorboard session at 'logging_folder_fullpath' to see progress
