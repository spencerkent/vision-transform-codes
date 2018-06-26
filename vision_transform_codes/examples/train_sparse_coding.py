"""
Train a sparse coding dictionary on the Field natural images dataset
"""
import sys
sys.path.insert(0, '/home/spencerkent/Projects/vision-transform-codes/vision_transform_codes/')

import pickle
import numpy as np
from matplotlib import pyplot as plt
import torch

from training.sparse_coding import train_dictionary as sc_train
from utils.plotting import TrainingLivePlot
from utils.image_processing import create_patch_training_set

RUN_IDENTIFIER = 'test_sparse_coding'

BATCH_SIZE = 250
NUM_BATCHES = 4000  # 1 million patches total
PATCH_HEIGHT = 16
PATCH_WIDTH = 16

CODE_SIZE = 256
NUM_EPOCHS = 30

SC_PARAMS = {
    'num_epochs': NUM_EPOCHS,
    'code_inference_algorithm': 'ista',
    'inference_param_schedule': {
      0: {'sparsity_weight': 0.1, 'num_iters': 1000}},
    'dictionary_update_algorithm': 'sc_cheap_quadratic_descent',
    'dict_update_param_schedule': {
      0: {'stepsize': 0.005, 'num_iters': 1}},
    'checkpoint_schedule': {
      'checkpoint_folder_fullpath': '/media/expansion1/spencerkent/logfiles/vision_transform_codes/' + RUN_IDENTIFIER,
      NUM_BATCHES: None, 10*NUM_BATCHES: None, 20*NUM_BATCHES:None},
    'training_visualization_schedule': {
      0: None, 1000: None, 2000: None, 4000: None, 6000: None, 10000: None,
      20*NUM_BATCHES: None, 29*NUM_BATCHES: None}}

torch_device = torch.device('cuda:1')
torch.cuda.set_device(1)
# otherwise can put on 'cuda:0' or 'cpu'

# manually create large training set with one million whitened patches
one_mil_image_patches = create_patch_training_set(
    ['whiten_center_surround', 'patch'], (PATCH_HEIGHT, PATCH_WIDTH),
    BATCH_SIZE, NUM_BATCHES, edge_buffer=5, dataset='Field_NW_unwhitened',
    datasetparams={'exclude': []})

pickle.dump(one_mil_image_patches, open('/media/expansion1/spencerkent/Datasets/Field_natural_images/one_million_patches_whitened_June25.p', 'wb'))

# one_mil_image_patches = pickle.load(open(
#     '/media/expansion1/spencerkent/Datasets/Field_natural_images/one_million_patches_whitened_June25.p', 'rb')).astype('float32')

# send ALL image patches to the GPU
image_patches_gpu = torch.from_numpy(one_mil_image_patches).to(torch_device)

# create the dictionary Tensor on the GPU
sparse_coding_dictionary = torch.randn((PATCH_HEIGHT*PATCH_WIDTH, CODE_SIZE),
                                       device=torch_device)
sparse_coding_dictionary.div_(sparse_coding_dictionary.norm(p=2, dim=0))

# Create the LivePlot object
liveplot_obj = TrainingLivePlot(
    dict_plot_params={'total_num': CODE_SIZE, 'img_height': PATCH_HEIGHT,
                      'img_width': PATCH_WIDTH, 'plot_width': 16,
                      'plot_height': 16, 'renorm imgs': True},
    code_plot_params={'size': CODE_SIZE})

SC_PARAMS['training_visualization_schedule']['liveplot_object_reference'] = liveplot_obj

print("Here we go!")
sc_train(image_patches_gpu, sparse_coding_dictionary, SC_PARAMS)
