"""
Train a sparse coding dictionary with a topographic LSM prior

The {L}aplacian {S}cale {M}ixture variant of sparse coding parameterizes the
scale of the laplacian prior on each coefficient. This makes the prior on this
parameter a "hyperprior". We can further impose a topography on these scale
parameters, inducing a topography on the coefficients
"""
import _set_the_path

import math
import pickle
import numpy as np
import torch

from training.sparse_coding import train_dictionary
from utils.dataset_generation import create_patch_training_set, OneOutputDset
from utils import defaults
from utils import topographic as topo_utils

RUN_IDENTIFIER = 'test_lsm_topographic_sparse_coding'
LOGS_STORED_HERE = defaults.logging_directory

TRAINING_SET_SIZE = 1000000
VALIDATION_SET_SIZE = 10000
BATCH_SIZE = 250
PATCH_HEIGHT = 16
PATCH_WIDTH = 16

# Define a topography on the scale (AKA inverse-rate) parameters
SHAPE_OF_TOPOGRAPHY = [8, 8]
TOPOGRAPHY_NEIGHBORHOOD_SIZES = [3, 3]
# ^each scale parameter projects to 3 coefficients in each dimension, a total
#  of 9 'children' coefficient
TOPOGRAPHY_STRIDES = [2, 2]
# ^The stride between scale parameters is 2, so the neighborhoods overlap by
#  one coefficient. These coefficients average the input from their two
#  'parents.'

# The connectivity matrix is used for fast updates to the rate/scale parameters
# during LSM inference. See utils/topographic.py and
# analysis_transforms/*/lsm_reweighted_ista_fista.py for more details
connectivity_mat = topo_utils.generate_LSM_topo_connectivity_matrix(
    SHAPE_OF_TOPOGRAPHY, TOPOGRAPHY_NEIGHBORHOOD_SIZES, TOPOGRAPHY_STRIDES)
# the spreading matrix is a normalized version of the transpose connectivity
# matrix. It's useful for spreading the hierarchical rate/scale variables
# to all the coefficients
spreading = (connectivity_mat / np.sum(connectivity_mat, axis=1)[:, None]).T
# for each rate/scale variable it's neighborhood has 3x3 elements:
neighborhood_sizes = (
    np.ones([1, np.prod(SHAPE_OF_TOPOGRAPHY)], dtype='float32') *
    np.prod(TOPOGRAPHY_NEIGHBORHOOD_SIZES))


# the topography of the rate/scale variables determines the topography of the
# cofficients themselves
coeff_topo_shape = topo_utils.compute_coeff_topo_size(
    SHAPE_OF_TOPOGRAPHY, TOPOGRAPHY_STRIDES)

# codes during inference are a flattened version of this topography
flattened_code_size = np.prod(coeff_topo_shape)
NUM_EPOCHS = 10
iters_per_epoch = int(math.ceil(TRAINING_SET_SIZE / BATCH_SIZE))


LOAD_DATA_FROM_DISK = False
if not LOAD_DATA_FROM_DISK:
  SAVE_TRAINING_DATA_FOR_FASTER_RELAUNCH = True
  print('Creating training and validation datasets...')
  trn_val_dsets = {
      'training': create_patch_training_set(
        num_samples=TRAINING_SET_SIZE,
        patch_dimensions=(PATCH_HEIGHT, PATCH_WIDTH), edge_buffer=5,
        dataset='Field_NW', order_of_preproc_ops=[
          'standardize_data_range', 'whiten_center_surround', 'patch']),
      'validation': create_patch_training_set(
        num_samples=VALIDATION_SET_SIZE,
        patch_dimensions=(PATCH_HEIGHT, PATCH_WIDTH), edge_buffer=5,
        dataset='Field_NW', order_of_preproc_ops=[
          'standardize_data_range', 'whiten_center_surround', 'patch'])}
  if SAVE_TRAINING_DATA_FOR_FASTER_RELAUNCH:
    pickle.dump(trn_val_dsets, open(defaults.dataset_directory /
      'Field_natural_images/patches_sdr_white.p', 'wb'))
else:
  trn_val_dsets = pickle.load(open(defaults.dataset_directory /
    'Field_natural_images/patches_sdr_white.p', 'rb'))


torch_device = torch.device('cuda:0')

lsm_params = {
    'beta': 0.00001,
    'update_interval': 10,
    'lsm_rate_var_structure': 'topographic',
    'lsm_rate_struct_params': {
      'connectivity_matrix': torch.from_numpy(connectivity_mat).to(torch_device),
      'spreading_matrix': torch.from_numpy(spreading).to(torch_device),
      'neighborhood_sizes': torch.from_numpy(neighborhood_sizes).to(torch_device),
      'coeff_topo_shape': coeff_topo_shape}}

SC_PARAMS = {
    'mode': 'fully-connected',
    'num_epochs': NUM_EPOCHS,
    'code_inference_algorithm': 'lsm_fista',
    'lsm_parameters': lsm_params,
    'inference_param_schedule': {
      0: {'sparsity_weight': 0.00003, 'num_iters': 25},
      2*iters_per_epoch: {'sparsity_weight': 0.00003, 'num_iters': 50},
      5*iters_per_epoch: {'sparsity_weight': 0.00003, 'num_iters': 100}},
    'dictionary_update_algorithm': 'sc_cheap_quadratic_descent',
    'dict_update_param_schedule': {
      0: {'stepsize': 0.2, 'num_iters': 1},
      5*iters_per_epoch: {'stepsize': 0.1, 'num_iters': 1}},
    # write various tensorboard logs on the following schedule:
    'training_visualization_schedule': set([0, 500, 1000, 2000]),
    'reshaped_kernel_size': (PATCH_HEIGHT, PATCH_WIDTH),
    # actually store all logs here:
    'logging_folder_fullpath': LOGS_STORED_HERE / RUN_IDENTIFIER,
    # checkpoint the dictionary on this interval
    'checkpoint_schedule': set([iters_per_epoch,
                                (NUM_EPOCHS*iters_per_epoch)-1])}
SC_PARAMS['training_visualization_schedule'].update(set(
    [iters_per_epoch*x for x in range(1, NUM_EPOCHS)]))


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
sparse_coding_dictionary = torch.randn((flattened_code_size,
  PATCH_HEIGHT*PATCH_WIDTH),
                                       device=torch_device)
# start out the dictionaries with norm 1
sparse_coding_dictionary.div_(
    sparse_coding_dictionary.norm(p=2, dim=1)[:, None])

train_dictionary(image_patches_gpu_training, image_patches_gpu_validation,
                 sparse_coding_dictionary, SC_PARAMS)
