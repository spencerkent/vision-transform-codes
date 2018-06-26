"""
Train an ICA dictionary on the Field natural images dataset
"""
import sys
sys.path.insert(0, '/home/spencerkent/Projects/vision-transform-codes/vision_transform_codes/')

import pickle
import numpy as np
from matplotlib import pyplot as plt
import torch

from training.ica import train_dictionary as ica_train
from utils.plotting import TrainingLivePlot
from utils.image_processing import create_patch_training_set

RUN_IDENTIFIER = 'test_ICA'

BATCH_SIZE = 250
NUM_BATCHES = 4000  # 1 million patches total
PATCH_HEIGHT = 16
PATCH_WIDTH = 16

CODE_SIZE = PATCH_HEIGHT * PATCH_WIDTH
NUM_EPOCHS = 30

ICA_PARAMS = {
    'num_epochs': NUM_EPOCHS,
    'dictionary_update_algorithm': 'ica_natural_gradient',
    'dict_update_param_schedule': {
      0: {'stepsize': 0.01, 'num_iters': 1},
      5 * NUM_BATCHES: {'stepsize': 0.0005, 'num_iters': 1},
      15 * NUM_BATCHES: {'stepsize': 0.0001, 'num_iters': 1}},
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
Q, R = np.linalg.qr(np.random.standard_normal((PATCH_HEIGHT*PATCH_WIDTH,
                                               CODE_SIZE)))
ica_dictionary = torch.from_numpy(Q.astype('float32')).to(torch_device)

# Create the LivePlot object
liveplot_obj = TrainingLivePlot(
    dict_plot_params={'total_num': CODE_SIZE, 'img_height': PATCH_HEIGHT,
                      'img_width': PATCH_WIDTH, 'plot_width': 16,
                      'plot_height': 16, 'renorm imgs': True},
    code_plot_params={'size': CODE_SIZE})

ICA_PARAMS['training_visualization_schedule']['liveplot_object_reference'] = liveplot_obj

print("Here we go!")
ica_train(image_patches_gpu, ica_dictionary, ICA_PARAMS)
