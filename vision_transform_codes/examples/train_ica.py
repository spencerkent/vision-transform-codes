"""
Train an ICA dictionary on the Field natural images dataset
"""
import sys
sys.path.insert(0, '/home/spencerkent/Projects/vision-transform-codes/vision_transform_codes/')

import pickle
import numpy as np
from matplotlib import pyplot as plt
import torch

from utils.plotting import TrainingLivePlot
from ica import train_dictionary as ica_train

RUN_IDENTIFIER = 'test_ICA'
BATCH_SIZE = 250
PATCH_FLATSIZE = 256
CODE_SIZE = PATCH_FLATSIZE
NUM_EPOCHS = 30

ICA_PARAMS = {
    'num_epochs': NUM_EPOCHS,
    'dictionary_update_algorithm': 'ica_natural_gradient',
    'dict_update_param_schedule': {
      0: {'stepsize': 0.01, 'num_iters': 1},
      5 * (1000000 // BATCH_SIZE): {'stepsize': 0.0005, 'num_iters': 1},
      15 * (1000000 // BATCH_SIZE): {'stepsize': 0.0001, 'num_iters': 1}},
    'checkpoint_schedule': {
      'checkpoint_folder_fullpath': '/media/expansion1/spencerkent/logfiles/vision_transform_codes/' + RUN_IDENTIFIER,
      1000000 // BATCH_SIZE: None, 10 * (1000000 // BATCH_SIZE): None,
      20 * (1000000 // BATCH_SIZE): None, 29 * (1000000 // BATCH_SIZE): None},
    'training_visualization_schedule': {
      0: None, 1000: None, 2000: None, 4000: None, 6000: None, 10000: None,
      20 * (1000000 // BATCH_SIZE): None, 29 * (1000000 // BATCH_SIZE): None}}

torch_device = torch.device('cuda:1')
torch.cuda.set_device(1)
# otherwise can put on 'cuda:0' or 'cpu'

field_precomputed_patches = pickle.load(open(
    '/media/expansion1/spencerkent/Datasets/Field_natural_images/one_million_patches_June19.p',
    'rb')).astype('float32')
# send ALL image patches to the GPU
one_mil_image_patches = torch.from_numpy(field_precomputed_patches).to(torch_device)

# create the dictionary Tensor on the GPU
Q, R = np.linalg.qr(np.random.standard_normal((PATCH_FLATSIZE, CODE_SIZE)))
ica_dictionary = torch.from_numpy(Q.astype('float32')).to(torch_device)

# Create the LivePlot object
liveplot_obj = TrainingLivePlot(
    dict_plot_params={'total_num': CODE_SIZE, 'img_height': 16, 'img_width': 16,
                      'plot_width': 16, 'plot_height': 16, 'renorm imgs':True},
    code_plot_params={'size': CODE_SIZE})

ICA_PARAMS['training_visualization_schedule']['liveplot_object_reference'] = liveplot_obj

print("Here we go!")
ica_train(one_mil_image_patches, ica_dictionary, ICA_PARAMS)
