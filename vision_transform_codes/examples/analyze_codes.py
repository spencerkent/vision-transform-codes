"""
Examples of some things we might want to do with the learned codes/dictionaries
"""
import sys
import os
examples_fullpath = os.path.dirname(os.path.abspath(__file__))
toplevel_dir_fullpath = examples_fullpath[:examples_fullpath.rfind('/')+1]
sys.path.insert(0, toplevel_dir_fullpath)

import argparse
import pickle
import numpy as np
from matplotlib import pyplot as plt
import scipy.io
import torch

from training.pca import train_dictionary as pca_train
from analysis_transforms import invertible_linear
from analysis_transforms import fista
from utils.plotting import display_dictionary
from utils.plotting import display_codes
from utils.plotting import display_code_marginal_densities
from utils.image_processing import patches_from_single_image
from utils.image_processing import assemble_image_from_patches
from utils.image_processing import whiten_center_surround
from utils.image_processing import unwhiten_center_surround
from utils.image_processing import subtract_patch_DC
from utils.quantization import compute_pSNR

patch_dim=(8, 8)

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_directory",
    help="The full path to the directory where checkpoints are stored")
parser.add_argument("dataset",
    help="Which dataset to grab test images from")
parser.add_argument("-i", "--training_iter_num", 
                    help="The iteration number of the checkpoint")
script_args = parser.parse_args()
if script_args.training_iter_num is None:
  # get the last checkpoint
  iter_nums = []
  for _, _, filenames in os.walk(script_args.checkpoint_directory):
    for filename in filenames:
      if filename[:27] == 'checkpoint_dictionary_iter_':
        iter_nums.append(int(filename[27:]))
    break
  dictionary = pickle.load(open(script_args.checkpoint_directory + 
    '/checkpoint_dictionary_iter_' + str(max(iter_nums)), 'rb'))
  print('checkpoint num: ', str(max(iter_nums)))
else:
  dictionary = pickle.load(open(script_args.checkpoint_directory + 
    '/checkpoint_dictionary_iter_' + script_args.training_iter_num, 'rb'))

if script_args.dataset == 'Field_NW_unwhitened':
  # specific images to visualize the results with
  test_images = scipy.io.loadmat('/media/expansion1/spencerkent/Datasets/Field_natural_images/unwhitened.mat')['IMAGESr'].astype('float32')[:, :, [5, 9]]
  patch_dataset = pickle.load(open(
      '/media/expansion1/spencerkent/Datasets/Field_natural_images/one_million_patches_November11.p', 'rb'))
  patches_test_im_1, p_pos_im_1 = patches_from_single_image(
      test_images[:, :, 0].astype('float32'), patch_dim)
  patches_test_im_2, p_pos_im_2 = patches_from_single_image(
      test_images[:, :, 1].astype('float32'), patch_dim)
  img_patch_comp_means = patch_dataset['original_patch_means']
  centered_patches_test_im_1 = patches_test_im_1 - img_patch_comp_means[:, None]
  centered_patches_test_im_2 = patches_test_im_2 - img_patch_comp_means[:, None]
elif script_args.dataset == 'Kodak_whitened':
  patch_dataset = pickle.load(open(
      '/media/expansion1/spencerkent/Datasets/Kodak/kodak_patches_1mil_8x8_whitened_and_centered.p', 'rb'))
  # specific images to visualize the results with
  test_images = pickle.load(open('/media/expansion1/spencerkent/Datasets/Kodak/kodak_full_images_testing.p', 'rb'))
  patches_test_im_1, p_pos_im_1 = patches_from_single_image(
      whiten_center_surround(test_images[0].astype('float32')), patch_dim)
  patches_test_im_2, p_pos_im_2 = patches_from_single_image(
      whiten_center_surround(test_images[1].astype('float32')), patch_dim)
  img_patch_comp_means = patch_dataset['original_patch_means']
  centered_patches_test_im_1 = patches_test_im_1 - img_patch_comp_means[:, None]
  centered_patches_test_im_2 = patches_test_im_2 - img_patch_comp_means[:, None]

elif script_args.dataset == 'Kodak_unwhitened':
  patch_dataset = pickle.load(open(
      '/media/expansion1/spencerkent/Datasets/Kodak/kodak_patches_1mil_8x8_centerd_on_zero.p', 'rb'))
  # specific images to visualize the results with
  test_images = pickle.load(open('/media/expansion1/spencerkent/Datasets/Kodak/kodak_full_images_testing.p', 'rb'))
  patches_test_im_1, p_pos_im_1 = patches_from_single_image(
      test_images[0].astype('float32'), patch_dim)
  patches_test_im_2, p_pos_im_2 = patches_from_single_image(
      test_images[1].astype('float32'), patch_dim)
  img_patch_comp_means = patch_dataset['original_patch_means']
  centered_patches_test_im_1 = patches_test_im_1 - img_patch_comp_means[:, None]
  centered_patches_test_im_2 = patches_test_im_2 - img_patch_comp_means[:, None]

elif script_args.dataset == 'Kodak_dc_centered':
  patch_dataset = pickle.load(open(
      '/media/expansion1/spencerkent/Datasets/Kodak/kodak_patches_1mil_8x8_centerd_and_dc_subtracted.p', 'rb'))
  # specific images to visualize the results with
  test_images = pickle.load(open('/media/expansion1/spencerkent/Datasets/Kodak/kodak_full_images_testing.p', 'rb'))
  patches_test_im_1, p_pos_im_1 = patches_from_single_image(
      test_images[0].astype('float32'), patch_dim)
  patches_test_im_2, p_pos_im_2 = patches_from_single_image(
      test_images[1].astype('float32'), patch_dim)
  img_patch_comp_means = patch_dataset['original_component_means']
  centered_patches_test_im_1, patch_means_im_1 = \
      subtract_patch_DC(patches_test_im_1 - img_patch_comp_means[:, None])
  centered_patches_test_im_2, patch_means_im_2 = \
      subtract_patch_DC(patches_test_im_2 - img_patch_comp_means[:, None])

# print(np.mean(centered_patches_test_im_1, axis=0)[0:100])
# print(np.mean(centered_patches_test_im_1, axis=1))
# input('Wait')

torch_device = torch.device('cuda:1')
torch.cuda.set_device(1)
# codes_training = fista.run(
#     torch.from_numpy(zero_mean_patches).to(torch_device),
#     torch.from_numpy(dictionary).to(torch_device), 0.1, 1000).cpu().numpy()

codes_test_im_1 = fista.run(
    torch.from_numpy(centered_patches_test_im_1).to(torch_device),
    torch.from_numpy(dictionary).to(torch_device), 5.0, 600).cpu().numpy()

codes_test_im_2 = fista.run(
    torch.from_numpy(centered_patches_test_im_2).to(torch_device),
    torch.from_numpy(dictionary).to(torch_device), 5.0, 600).cpu().numpy()

reconned_patches_im_1 = (np.dot(dictionary, codes_test_im_1) + img_patch_comp_means[:, None])
reconned_patches_im_2 = (np.dot(dictionary, codes_test_im_2) + img_patch_comp_means[:, None])


reconned_im_1 = assemble_image_from_patches(reconned_patches_im_1, patch_dim,
                                            p_pos_im_1)
reconned_im_2 = assemble_image_from_patches(reconned_patches_im_2, patch_dim,
                                            p_pos_im_2)
dict_plots = display_dictionary(dictionary, patch_dim,
      'Trained dictionary elements', renormalize=False)

code_samp_im1 = display_codes(codes_test_im_1[:, np.random.choice(np.arange(codes_test_im_1.shape[1]), 20, replace=False)], 'Example of codes for image 1')
code_samp_im2 = display_codes(codes_test_im_2[:, np.random.choice(np.arange(codes_test_im_2.shape[1]), 20, replace=False)], 'Example of codes for image 2')
# code_marg_1 = display_code_marginal_densities(codes_training[:, 0:100000], 100,
#                                               'marginal code densities for image 1')

fig = plt.figure(figsize=(15, 15))
fig.suptitle('Test image 1')
plt.subplot(1, 2, 1)
plt.imshow(test_images[0], cmap='Greys_r')
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(reconned_im_1, cmap='Greys_r')
plt.title('Reconstructed')

fig = plt.figure(figsize=(15, 15))
fig.suptitle('Test image 2')
plt.subplot(1, 2, 1)
plt.imshow(test_images[1], cmap='Greys_r')
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(reconned_im_2, cmap='Greys_r')
plt.title('Reconstructed')

plt.show()
