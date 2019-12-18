"""
Example: Train a sparse coding dictionary. These settings for Field natural
images dset
"""
import _set_the_path

import argparse
import pickle
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import torch

from training.sparse_coding import train_dictionary as sc_train
from utils.dataset_generation import create_patch_training_set

RUN_IDENTIFIER = 'test_sparse_coding'

BATCH_SIZE = 250
NUM_BATCHES = 4000  # 1 million patches total
PATCH_HEIGHT = 16
PATCH_WIDTH = 16

CODE_SIZE = 1 * PATCH_HEIGHT*PATCH_WIDTH  # critically sampled
NUM_EPOCHS = 10

SC_PARAMS = {
    'mode': 'fully-connected',
    'num_epochs': NUM_EPOCHS,
    'code_inference_algorithm': 'fista',
    'inference_param_schedule': {
      0: {'sparsity_weight': 0.1, 'num_iters': 50},
      10*NUM_BATCHES: {'sparsity_weight': 0.1, 'num_iters': 100},
      20*NUM_BATCHES: {'sparsity_weight': 0.1, 'num_iters': 200}},
    'dictionary_update_algorithm': 'sc_cheap_quadratic_descent',
    'dict_update_param_schedule': {
      0: {'stepsize': 0.05, 'num_iters': 1},
      10*NUM_BATCHES: {'stepsize': 0.01, 'num_iters': 1},
      20*NUM_BATCHES: {'stepsize': 0.005, 'num_iters': 1}},
    'training_visualization_schedule': {
      0: None, 1000: None, 2000: None, 4000: None, 8000: None, 
      (NUM_EPOCHS * NUM_BATCHES) - 1: None,
      'reshaped_kernel_size': (PATCH_HEIGHT, PATCH_WIDTH)}}
SC_PARAMS['training_visualization_schedule'].update(
    {NUM_BATCHES*x: None for x in range(NUM_EPOCHS)})

# Arguments for dataset and logging
parser = argparse.ArgumentParser()
parser.add_argument("data_id",
    help="Name of the dataset (currently allowable: " +
         "Field_NW, vanHateren, Kodak_BW)")
parser.add_argument("data_filepath", help="The full path to dataset on disk")
parser.add_argument("-l", "--logfile_dir",
                    help="Optionally checkpoint the model here")
script_args = parser.parse_args()

if script_args.logfile_dir is not None:
  SC_PARAMS['logging_folder_fullpath'] = Path(script_args.logfile_dir) / RUN_IDENTIFIER
  SC_PARAMS['checkpoint_schedule'] = {
      NUM_BATCHES: None, (NUM_EPOCHS*NUM_BATCHES)-1: None}

torch_device = torch.device('cuda:1')
torch.cuda.set_device(1)
# otherwise can put on 'cuda:0' or 'cpu'

# manually create large training set with one million whitened patches
patch_dataset = create_patch_training_set(
    ['whiten_center_surround', 'patch', 'center_each_component'],
    (PATCH_HEIGHT, PATCH_WIDTH), BATCH_SIZE, NUM_BATCHES ,
    edge_buffer=5, dataset=script_args.data_id,
    datasetparams={'filepath': script_args.data_filepath,
                   'exclude': []})
#################################################################
# save these to disk if you want always train on the same patches
# or if you want to speed things up in the future
#################################################################
# pickle.dump(one_mil_image_patches, open('/media/expansion1/spencerkent/Datasets/Field_natural_images/one_million_patches_November11.p', 'wb'))

# patch_dataset = pickle.load(open('/media/expansion1/spencerkent/Datasets/Field_natural_images/one_million_patches_November11.p', 'wb'))

# send ALL image patches to the GPU
image_patches_gpu = torch.from_numpy(
    patch_dataset['batched_patches']).to(torch_device)

# create the dictionary Tensor on the GPU
sparse_coding_dictionary = torch.randn((PATCH_HEIGHT*PATCH_WIDTH, CODE_SIZE),
                                       device=torch_device)
sparse_coding_dictionary.div_(sparse_coding_dictionary.norm(p=2, dim=0))

print("Here we go!")
sc_train(image_patches_gpu, sparse_coding_dictionary, SC_PARAMS)
