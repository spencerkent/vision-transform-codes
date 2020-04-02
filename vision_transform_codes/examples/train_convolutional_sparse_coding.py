"""
Train a *convolutional* sparse coding dictionary.

This shows some viable parameter settings for David Field's natural images
dataset. So long as you use the 'standardize_data_range' preprocessing step,
these should also work well for other datasets too
(see utils/dataset_generation.py).
"""
import _set_the_path

import pickle
import torch

from training.sparse_coding import train_dictionary
from utils.convolutions import get_padding_amt
from utils.dataset_generation import create_patch_training_set
from utils import defaults

RUN_IDENTIFIER = 'test_convolutional_sparse_coding'
LOGS_STORED_HERE = defaults.logging_directory

PATCH_HEIGHT = 256
PATCH_WIDTH = 256
BATCH_SIZE = 5  # (PATCH_HEIGHT x PATCH_WIDTH) images
NUM_BATCHES = 2000  # 10000 images in an EPOCH
VAL_SET_SIZE = 50 # (PATCH_HEIGHT x PATCH_WIDTH) images

KERNEL_HEIGHT = 16
KERNEL_WIDTH = 16
KERNEL_STRIDE_VERT = 8
KERNEL_STRIDE_HORZ = 8
assert KERNEL_STRIDE_HORZ <= KERNEL_WIDTH
assert KERNEL_STRIDE_VERT <= KERNEL_HEIGHT
assert KERNEL_WIDTH % KERNEL_STRIDE_HORZ == 0
assert KERNEL_HEIGHT % KERNEL_STRIDE_VERT == 0

CODE_SIZE = 64  # The stride makes this critically sampled
NUM_EPOCHS = 10
iters_per_epoch = NUM_BATCHES

# Dataset generation/loading
vert_padding = get_padding_amt(PATCH_HEIGHT, KERNEL_HEIGHT,
                               KERNEL_STRIDE_VERT)
horz_padding = get_padding_amt(PATCH_WIDTH, KERNEL_WIDTH,
                               KERNEL_STRIDE_HORZ)
LOAD_DATA_FROM_DISK = False
if not LOAD_DATA_FROM_DISK:
  SAVE_TRAINING_DATA_FOR_FASTER_RELAUNCH = True
  print('Creating training and validation datasets...')
  trn_val_dsets = {
      'training': create_patch_training_set(
        num_batches=NUM_BATCHES, batch_size=BATCH_SIZE,
        patch_dimensions=(PATCH_HEIGHT, PATCH_WIDTH), edge_buffer=5,
        dataset='Field_NW', order_of_preproc_ops=[
          'standardize_data_range', 'whiten_center_surround', 'patch'],
        extra_params={'padding': (vert_padding, horz_padding),
                      'flatten_patches': False}),
      'validation': create_patch_training_set(
        num_batches=1, batch_size=VAL_SET_SIZE,
        patch_dimensions=(PATCH_HEIGHT, PATCH_WIDTH), edge_buffer=5,
        dataset='Field_NW', order_of_preproc_ops=[
          'standardize_data_range', 'whiten_center_surround', 'patch'],
        extra_params={'padding': (vert_padding, horz_padding),
                      'flatten_patches': False})}
  if SAVE_TRAINING_DATA_FOR_FASTER_RELAUNCH:
    pickle.dump(trn_val_dsets, open(defaults.dataset_directory /
      'Field_natural_images/conv_patches_sdr_white.p', 'wb'))
else:
  trn_val_dsets = pickle.load(open(defaults.dataset_directory /
    'Field_natural_images/conv_patches_sdr_white.p', 'rb'))

SC_PARAMS = {
    'mode': 'convolutional',
    'num_epochs': NUM_EPOCHS,
    'strides': (KERNEL_STRIDE_VERT, KERNEL_STRIDE_HORZ),
    'padding': (vert_padding, horz_padding),
    'code_inference_algorithm': 'ista',
    'inference_param_schedule': {
      0: {'sparsity_weight': 0.02, 'num_iters': 25},
      2*iters_per_epoch: {'sparsity_weight': 0.02, 'num_iters': 50},
      5*iters_per_epoch: {'sparsity_weight': 0.02, 'num_iters': 100}},
    'dictionary_update_algorithm': 'sc_cheap_quadratic_descent',
    'dict_update_param_schedule': {
      0: {'stepsize': 0.005, 'num_iters': 1},
      5*NUM_BATCHES: {'stepsize': 0.001, 'num_iters': 1},
      20*NUM_BATCHES: {'stepsize': 0.0001, 'num_iters': 1}},
    # write various tensorboard logs on the following schedule:
    'training_visualization_schedule': set([0, 500, 1000, 2000]),
    # actually store all logs here:
    'logging_folder_fullpath': LOGS_STORED_HERE / RUN_IDENTIFIER,
    # checkpoint the dictionary on this interval
    'checkpoint_schedule': set([iters_per_epoch,
                                (NUM_EPOCHS*iters_per_epoch)-1])}
SC_PARAMS['training_visualization_schedule'].update(set(
    [iters_per_epoch*x for x in range(1, NUM_EPOCHS)]))


# Now initialize modela and begin training
torch_device = torch.device('cuda:1')
torch.cuda.set_device(1)
# otherwise can put on 'cuda:0' or 'cpu'

# send ALL image patches to the GPU
image_patches_gpu_training = torch.from_numpy(
    trn_val_dsets['training']['batched_patches']).to(torch_device)
image_patches_gpu_validation = torch.from_numpy(
    trn_val_dsets['validation']['batched_patches']).to(torch_device)

# create the dictionary Tensor on the GPU
sparse_coding_dictionary = torch.randn(
    (CODE_SIZE, 1, KERNEL_HEIGHT, KERNEL_WIDTH), device=torch_device)
# start out the dictionaries with norm 1
sparse_coding_dictionary.div_(torch.squeeze(sparse_coding_dictionary.norm(
  p=2, dim=(1, 2, 3)))[:, None, None, None])

print("Here we go!")
train_dictionary(image_patches_gpu_training, image_patches_gpu_validation,
                 sparse_coding_dictionary, SC_PARAMS)
# ^point a tensorboard session at 'logging_folder_fullpath' to see progress
