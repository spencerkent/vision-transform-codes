"""
Test: Sparse coding, convolutional, ista, cheap quadratic descent
"""
import _set_the_path

import math
import pickle
import torch

from training.sparse_coding import train_dictionary
from utils.dataset_generation import OneOutputDset
from utils import defaults

RUN_IDENTIFIER = '_testing_sc_4'
LOGS_STORED_HERE = defaults.logging_directory

TRAINING_SET_SIZE = 20
VALIDATION_SET_SIZE = 10
BATCH_SIZE = 2
PATCH_HEIGHT = 256
PATCH_WIDTH = 256

KERNEL_HEIGHT = 16
KERNEL_WIDTH = 16
KERNEL_STRIDE_VERT = 8
KERNEL_STRIDE_HORZ = 8

CODE_SIZE = KERNEL_HEIGHT*KERNEL_WIDTH // 4 # critically sampled
NUM_EPOCHS = 1
iters_per_epoch = int(math.ceil(TRAINING_SET_SIZE / BATCH_SIZE))


trn_val_dsets = pickle.load(open(defaults.dataset_directory /
  'vtc_testing/field_white_256x256_w_padding.p', 'rb'))

SC_PARAMS = {
    'mode': 'convolutional',
    'num_epochs': NUM_EPOCHS,
    'strides': (KERNEL_STRIDE_VERT, KERNEL_STRIDE_HORZ),
    'padding': ((8, 8), (8, 8)),
    'code_inference_algorithm': 'ista',
    'inference_param_schedule': {
      0: {'sparsity_weight': 0.008, 'num_iters': 5}},
    'dictionary_update_algorithm': 'sc_cheap_quadratic_descent',
    'dict_update_param_schedule': {
      0: {'stepsize': 0.1, 'num_iters': 1},
      5*iters_per_epoch: {'stepsize': 0.05, 'num_iters': 1}},
    # write various tensorboard logs on the following schedule:
    'training_visualization_schedule': set([0, 10, 500]),
    'reshaped_kernel_size': (PATCH_HEIGHT, PATCH_WIDTH),
    # actually store all logs here:
    'logging_folder_fullpath': LOGS_STORED_HERE / RUN_IDENTIFIER,
    # checkpoint the dictionary on this interval
    'checkpoint_schedule': set([iters_per_epoch,
                                (NUM_EPOCHS*iters_per_epoch)-1])}
SC_PARAMS['training_visualization_schedule'].update(set(
    [iters_per_epoch*x for x in range(1, NUM_EPOCHS)]))


# Now initialize model and begin training
torch_device = torch.device('cuda:1')
# otherwise can put on 'cuda:0' or 'cpu'

# send ALL image patches to the GPU and wrap in a simple dataloader
image_patches_gpu_training = torch.utils.data.DataLoader(
    OneOutputDset(torch.from_numpy(
    trn_val_dsets['training']['patches']).to(torch_device)),
    batch_size=BATCH_SIZE, shuffle=True)
image_patches_gpu_validation = torch.utils.data.DataLoader(
    OneOutputDset(torch.from_numpy(
    trn_val_dsets['validation']['patches']).to(torch_device)),
    batch_size=BATCH_SIZE*10)  # larger batches for validation data
# if data is too big to all fit on GPU, just omit .to(torch_device) above.
# Can also add num_workers=x to the DataLoader constructor

# create the dictionary Tensor on the GPU
sparse_coding_dictionary = torch.randn(
    (CODE_SIZE, 1, KERNEL_HEIGHT, KERNEL_WIDTH), device=torch_device)
# start out the dictionaries with norm 1
sparse_coding_dictionary.div_(torch.squeeze(sparse_coding_dictionary.norm(
  p=2, dim=(1, 2, 3)))[:, None, None, None])

train_dictionary(image_patches_gpu_training, image_patches_gpu_validation,
                 sparse_coding_dictionary, SC_PARAMS)
