"""
Train a sparse coding dictionary with a subspace (group l2) penalty
"""
import _set_the_path

import math
import pickle
import numpy as np
import torch

from training.sparse_coding import train_dictionary
from utils.dataset_generation import create_patch_training_set, OneOutputDset
from utils import defaults

RUN_IDENTIFIER = 'test_subspace_sparse_coding'
LOGS_STORED_HERE = defaults.logging_directory

TRAINING_SET_SIZE = 1000000
VALIDATION_SET_SIZE = 10000
BATCH_SIZE = 250
PATCH_HEIGHT = 16
PATCH_WIDTH = 16

SUBSPACE_SIZE = 3
NUM_SUBSPACES = int(math.ceil((PATCH_HEIGHT*PATCH_WIDTH / SUBSPACE_SIZE)))
# try to make it as close to critically-sampled as possible

CODE_SIZE = SUBSPACE_SIZE * NUM_SUBSPACES
group_assignments = np.split(np.arange(CODE_SIZE), CODE_SIZE//SUBSPACE_SIZE)

NUM_EPOCHS = 10
iters_per_epoch = int(math.ceil(TRAINING_SET_SIZE / BATCH_SIZE))

torch_device = torch.device('cuda:0')

LOAD_DATA_FROM_DISK = True
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

# really simple connectivity matrix
conn_mat = np.zeros([CODE_SIZE, NUM_SUBSPACES], dtype='float32')
for g_idx in range(NUM_SUBSPACES):
  conn_mat[group_assignments[g_idx], g_idx] = 1
conn_mat_torch = torch.from_numpy(conn_mat).to(torch_device)

spw = 0.06 # sparsity weight
dps={0: {'stepsize': 0.5, 'num_iters': 1},
     2*iters_per_epoch: {'stepsize': 0.25, 'num_iters': 1},
     10*iters_per_epoch: {'stepsize': 0.1, 'num_iters': 1},
     15*iters_per_epoch: {'stepsize': 0.05, 'num_iters': 1}}


subspace_params = {
    'group_assignments': group_assignments,
    'connectivity_matrix': conn_mat_torch,
    'subspace_alignment_penalty': 0.0001}

SC_PARAMS = {
    'mode': 'fully-connected',
    'num_epochs': NUM_EPOCHS,
    'code_inference_algorithm': 'subspace_fista',
    'subspace_parameters': subspace_params,
    'inference_param_schedule': {
      0: {'sparsity_weight': spw, 'num_iters': 25},
      2*iters_per_epoch: {'sparsity_weight': spw, 'num_iters': 50},
      5*iters_per_epoch: {'sparsity_weight': spw, 'num_iters': 100}},
    'dictionary_update_algorithm': 'subspace_sc_cheap_quadratic_descent',
    'dict_update_param_schedule': dps,
    'dict_element_reset_schedule': {
      2*iters_per_epoch: {'filter_type': 'nonuniformity_within_group',
          'filter_params': {'num_gc_in_average': 20},
          'action': 'reset'},
      6*iters_per_epoch: {'filter_type': 'nonuniformity_within_group',
          'filter_params': {'num_gc_in_average': 20},
          'action': 'reset'},
      10*iters_per_epoch: {'filter_type': 'cosine_sim_threshold',
        'filter_params': {'cue_user': False, 'threshold': 0.5,
                          'only_sim_within_group': True},
           'action': 'reset'}},
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
sparse_coding_dictionary = torch.randn((CODE_SIZE,
  PATCH_HEIGHT*PATCH_WIDTH),
                                       device=torch_device)
# start out the dictionaries with norm 1
sparse_coding_dictionary.div_(
    sparse_coding_dictionary.norm(p=2, dim=1)[:, None])

train_dictionary(image_patches_gpu_training, image_patches_gpu_validation,
                 sparse_coding_dictionary, SC_PARAMS)
