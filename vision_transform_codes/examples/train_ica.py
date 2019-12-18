"""
Example: Train an ICA dictionary. These settings for Field natural images dset
"""
import _set_the_path

import argparse
import pickle
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import torch

from training.ica import train_dictionary as ica_train
from utils.plotting import TrainingLivePlot
from utils.dataset_generation import create_patch_training_set

RUN_IDENTIFIER = 'test_ICA'

BATCH_SIZE = 250
NUM_BATCHES = 4000  # 1 million patches total
PATCH_HEIGHT = 16
PATCH_WIDTH = 16

CODE_SIZE = PATCH_HEIGHT * PATCH_WIDTH
NUM_EPOCHS = 10

ICA_PARAMS = {
    'num_epochs': NUM_EPOCHS,
    'dictionary_update_algorithm': 'ica_natural_gradient',
    'dict_update_param_schedule': {
      0: {'stepsize': 0.01, 'num_iters': 1},
      2 * NUM_BATCHES: {'stepsize': 0.005, 'num_iters': 1},
      5 * NUM_BATCHES: {'stepsize': 0.0025, 'num_iters': 1},
      7 * NUM_BATCHES: {'stepsize': 0.001, 'num_iters': 1}},
    'training_visualization_schedule': {
      0: None, 1000: None, 2000: None, 4000: None, 8000: None, 
      (NUM_EPOCHS * NUM_BATCHES) - 1: None,
      'reshaped_kernel_size': (PATCH_HEIGHT, PATCH_WIDTH)}}
ICA_PARAMS['training_visualization_schedule'].update(
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
  ICA_PARAMS['logging_folder_fullpath'] = Path(script_args.logfile_dir) / RUN_IDENTIFIER
  ICA_PARAMS['checkpoint_schedule'] = {
      NUM_BATCHES: None, (NUM_EPOCHS*NUM_BATCHES)-1: None}

torch_device = torch.device('cuda:1')
torch.cuda.set_device(1)
# otherwise can put on 'cuda:0' or 'cpu'

# manually create large training set with one million whitened patches
patch_dataset = create_patch_training_set(
    ['whiten_center_surround', 'patch', 'center_each_component'],
    (PATCH_HEIGHT, PATCH_WIDTH), BATCH_SIZE, NUM_BATCHES,
    edge_buffer=5, dataset=script_args.data_id,
    datasetparams={'filepath': script_args.data_filepath,
                   'exclude': []})

#################################################################
# save these to disk if you want always train on the same patches
# or if you want to speed things up in the future
#################################################################
# pickle.dump(patch_dataset, open('/media/expansion1/spencerkent/Datasets/Field_natural_images/one_million_patches_whitened_June25.p', 'wb'))

# patch_dataset = pickle.load(open(
#     '/media/expansion1/spencerkent/Datasets/Field_natural_images/one_million_patches_whitened_June25.p', 'rb')).astype('float32')

# send ALL image patches to the GPU
image_patches_gpu = torch.from_numpy(
    patch_dataset['batched_patches']).to(torch_device)

# create the dictionary Tensor on the GPU
Q, R = np.linalg.qr(np.random.standard_normal((PATCH_HEIGHT*PATCH_WIDTH,
                                               CODE_SIZE)))
ica_dictionary = torch.from_numpy(Q.astype('float32')).to(torch_device)

print("Here we go!")
ica_train(image_patches_gpu, ica_dictionary, ICA_PARAMS)
