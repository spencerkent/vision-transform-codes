"""
Train an PCA dictionary
"""
import _set_the_path

import pickle
from matplotlib import pyplot as plt
import torch

from training.pca import train_dictionary
from analysis_transforms.fully_connected import invertible_linear
from utils.dataset_generation import create_patch_training_set
from utils.plotting import display_dictionary
from utils import defaults
from utils.plotting import compute_pSNR

RUN_IDENTIFIER = 'test_PCA'

NUM_IMAGES_TRAIN = 1000000
PATCH_HEIGHT = 16
PATCH_WIDTH = 16

CODE_SIZE = PATCH_HEIGHT * PATCH_WIDTH

patch_dataset = create_patch_training_set(
    num_batches=1, batch_size=NUM_IMAGES_TRAIN,
    patch_dimensions=(PATCH_HEIGHT, PATCH_WIDTH), edge_buffer=5,
    dataset='Field_NW',
    order_of_preproc_ops=['standardize_data_range',
                          'patch', 'center_each_component'])

torch_device = torch.device('cuda:1')
torch.cuda.set_device(1)
# otherwise can put on 'cuda:0' or 'cpu'

# we are going to 'unbatch' them because pca will train on the whole dataset
# at once
image_patches_gpu = torch.from_numpy(
    patch_dataset['batched_patches'][0]).to(torch_device)

pca_dictionary = train_dictionary(image_patches_gpu)
codes = invertible_linear.run(image_patches_gpu, pca_dictionary, orthonormal=True)
reconstructions = torch.mm(codes, pca_dictionary)

plots = display_dictionary(pca_dictionary.cpu().numpy(),
    reshaping=(16, 16),
    plot_title='PCA-determined basis functions')

plt.show()
