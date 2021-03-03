"""
Test: Laplacian scale mixture sparse coding, disjoint group hyperprior
"""
import _set_the_path

import math
import pickle
import numpy as np
import torch

from training.sparse_coding import train_dictionary
from utils.dataset_generation import OneOutputDset
from utils import defaults
from utils import topographic as topo_utils

RUN_IDENTIFIER = '_testing_sc_7'
LOGS_STORED_HERE = defaults.logging_directory

TRAINING_SET_SIZE = 10000
VALIDATION_SET_SIZE = 5000
BATCH_SIZE = 1000
PATCH_HEIGHT = 16
PATCH_WIDTH = 16

# groups of size 4
GROUP_SIZE = 4
NUM_GROUPS = 6

CODE_SIZE = GROUP_SIZE * NUM_GROUPS

group_assignments = np.array_split(np.arange(CODE_SIZE), NUM_GROUPS)

# the corresponding connectivity matrix has scale parameters with a
# neighborhood of size 4 and a stride of 4:
connectivity_mat = topo_utils.generate_LSM_topo_connectivity_matrix(
    (NUM_GROUPS,), (GROUP_SIZE,), (GROUP_SIZE,))
spreading = (connectivity_mat / np.sum(connectivity_mat, axis=1)[:, None]).T
neighborhood_sizes = np.ones([1, NUM_GROUPS], dtype='float32') * GROUP_SIZE

coeff_topo = topo_utils.compute_coeff_topo_size((NUM_GROUPS,), (GROUP_SIZE,))

NUM_EPOCHS = 1
iters_per_epoch = int(math.ceil(TRAINING_SET_SIZE / BATCH_SIZE))

trn_val_dsets = pickle.load(open(defaults.dataset_directory /
  'vtc_testing/field_white_16x16.p', 'rb'))

torch_device = torch.device('cuda:0')

SC_PARAMS = {
    'mode': 'fully-connected',
    'num_epochs': NUM_EPOCHS,
    'code_inference_algorithm': 'lsm_fista',
    'lsm_parameters': {'beta': 0.0001, 'update_interval': 10,
                       'lsm_rate_var_structure': 'disjoint_groups',
                       'lsm_rate_struct_params': {
                         'connectivity_matrix': torch.from_numpy(
                           connectivity_mat).to(torch_device),
                         'spreading_matrix': torch.from_numpy(
                           spreading).to(torch_device),
                         'neighborhood_sizes': torch.from_numpy(
                           neighborhood_sizes).to(torch_device),
                         'coeff_topo_shape': coeff_topo,
                         'groups': group_assignments}},
    'inference_param_schedule': {
      0: {'sparsity_weight': 0.0005, 'num_iters': 50}},
    'dictionary_update_algorithm': 'sc_cheap_quadratic_descent',
    'dict_update_param_schedule': {
      0: {'stepsize': 0.1, 'num_iters': 1},
      5*iters_per_epoch: {'stepsize': 0.05, 'num_iters': 1}},
    'dict_element_rp_schedule': {
      3*iters_per_epoch: {'filter_type': 'random',
          'filter_params': {'num_to_modify': 50},
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
sparse_coding_dictionary = torch.randn((CODE_SIZE, PATCH_HEIGHT*PATCH_WIDTH),
                                       device=torch_device)
# start out the dictionaries with norm 1
sparse_coding_dictionary.div_(
    sparse_coding_dictionary.norm(p=2, dim=1)[:, None])

train_dictionary(image_patches_gpu_training, image_patches_gpu_validation,
                 sparse_coding_dictionary, SC_PARAMS)
