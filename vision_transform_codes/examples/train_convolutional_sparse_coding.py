"""
Example: Train a *convolutional* sparse coding dictionary.
These settings for Field natural images dset
"""
import _set_the_path

import argparse
import pickle
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection as mpl_pcollection
from matplotlib.patches import Rectangle as mpl_rectangle
import torch

from analysis_transforms.convolutional import ista
from training.sparse_coding import train_dictionary as sc_train
from utils import image_processing
from utils.convolutions import get_padding_amt
from utils.convolutions import code_dim_from_padded_img_dim
from utils.dataset_generation import create_patch_training_set

RUN_IDENTIFIER = 'test_convolutional_sparse_coding'

BATCH_SIZE = 5  # whole images
PATCH_HEIGHT = 256
PATCH_WIDTH = 256
EDGE_BUFFER = 5
NUM_BATCHES = 2000  # 10000 images in an EPOCH
KERNEL_HEIGHT = 16
KERNEL_WIDTH = 16
KERNEL_STRIDE_VERT = 8
KERNEL_STRIDE_HORZ = 8
assert KERNEL_STRIDE_HORZ <= KERNEL_WIDTH
assert KERNEL_STRIDE_VERT <= KERNEL_HEIGHT
assert KERNEL_WIDTH % KERNEL_STRIDE_HORZ == 0
assert KERNEL_HEIGHT % KERNEL_STRIDE_VERT == 0

CODE_SIZE = 128
NUM_EPOCHS = 10

torch_device = torch.device('cuda:0')
torch.cuda.set_device(0)

# Arguments for dataset and logging
parser = argparse.ArgumentParser()
parser.add_argument("data_id",
    help="Name of the dataset (currently allowable: " +
         "Field_NW, vanHateren, Kodak_BW)")
parser.add_argument("data_filepath", help="The full path to dataset on disk")
parser.add_argument("-l", "--logfile_dir",
                    help="Optionally checkpoint the model here")
script_args = parser.parse_args()

##########################################################################
# Dataset generation
#TODO: place this in the dataset_generation util. Leaving here temporarily
vert_padding = get_padding_amt(PATCH_HEIGHT, KERNEL_HEIGHT,
                               KERNEL_STRIDE_VERT)
horz_padding = get_padding_amt(PATCH_WIDTH, KERNEL_WIDTH,
                               KERNEL_STRIDE_HORZ)

# manually create large training set
patch_dataset = create_patch_training_set(
    ['divide_by_constant', 'whiten_center_surround', 'patch', 'pad'],
    (PATCH_HEIGHT, PATCH_WIDTH), BATCH_SIZE, NUM_BATCHES ,
    edge_buffer=5, dataset=script_args.data_id,
    datasetparams={'filepath': script_args.data_filepath,
                   'exclude': [],
                   'div_constant': 255.,
                   'padding': (vert_padding, horz_padding)},
    flatten_patches=False)

SC_PARAMS = {
    'mode': 'convolutional',
    'strides': (KERNEL_STRIDE_VERT, KERNEL_STRIDE_HORZ),
    'padding': (vert_padding, horz_padding),
    'num_epochs': NUM_EPOCHS,
    'code_inference_algorithm': 'ista',
    'inference_param_schedule': {
      0: {'sparsity_weight': 0.0005, 'num_iters': 50},
      10*NUM_BATCHES: {'sparsity_weight': 0.0005, 'num_iters': 100},
      20*NUM_BATCHES: {'sparsity_weight': 0.0005, 'num_iters': 200}},
    'dictionary_update_algorithm': 'sc_cheap_quadratic_descent',
    'dict_update_param_schedule': {
      0: {'stepsize': 0.005, 'num_iters': 1},
      5*NUM_BATCHES: {'stepsize': 0.001, 'num_iters': 1},
      20*NUM_BATCHES: {'stepsize': 0.0001, 'num_iters': 1}},
    'training_visualization_schedule': {0: None, 100: None, 200: None, 1000:None}}
SC_PARAMS['training_visualization_schedule'].update(
    {NUM_BATCHES*x: None for x in range(1, NUM_EPOCHS)})

SC_PARAMS['logging_folder_fullpath'] = Path(
    '/media/expansion1/spencerkent/logfiles/vision_transform_codes/') / RUN_IDENTIFIER
SC_PARAMS['checkpoint_schedule'] = {
    NUM_BATCHES: None, (NUM_EPOCHS*NUM_BATCHES)-1: None}

# create the dictionary Tensor on the GPU
sparse_coding_dictionary = torch.randn(
    (CODE_SIZE, 1, KERNEL_HEIGHT, KERNEL_WIDTH), device=torch_device)
sparse_coding_dictionary.div_(torch.squeeze(sparse_coding_dictionary.norm(
  p=2, dim=(1, 2, 3)))[:, None, None, None])

gpu_image_corpus = torch.from_numpy(patch_dataset['batched_patches']).to(torch_device)

print("Here we go!")
sc_train(gpu_image_corpus, sparse_coding_dictionary, SC_PARAMS)
