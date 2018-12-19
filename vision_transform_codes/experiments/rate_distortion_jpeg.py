"""
Pitting my hand-rolled JPEG implementation against ffmpeg's
"""
import sys
sys.path.insert(0, '/home/spencerkent/Projects/EE290T-quantized-sparse-codes/EE290T_quantized_sparse_codes/')

import os
import time
import pickle
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
import torch

from analysis_transforms import invertible_linear
from utils import image_processing
from utils import plotting
from utils import quantization
from utils.matrix_zigzag import zigzag
from utils.jpeg import get_jpeg_quant_hifi_binwidths

starttime = time.time()
patch_dimensions = (8, 8)
logfile_dir = '/media/expansion1/spencerkent/logfiles/EE290T_project/experiment_logs/JPEG/'

# specific images to visualize the results with
test_images = pickle.load(open('/media/expansion1/spencerkent/Datasets/Kodak/kodak_full_images_testing.p', 'rb'))
patches_test_im_1, p_pos_im_1 = image_processing.patches_from_single_image(
    test_images[0].astype('float32'), patch_dimensions)
patches_test_im_2, p_pos_im_2 = image_processing.patches_from_single_image(
    test_images[1].astype('float32'), patch_dimensions)

jpeg_hifi_binwidths = get_jpeg_quant_hifi_binwidths()
jpeg_quant_multipliers = [0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
dct_matrix_zigzag_order = pickle.load(open('/home/spencerkent/Projects/EE290T-quantized-sparse-codes/EE290T_quantized_sparse_codes/utils/dct_matrix_8x8_zigzag_ordering.p', 'rb')).astype('float32')
reordered_inds = zigzag(np.arange(64, dtype='int').reshape((8, 8))).astype('int')
dct_matrix = dct_matrix_zigzag_order[:, reordered_inds]

torch_device = torch.device('cuda:1')
torch.cuda.set_device(1)
torch_dct_matrix = torch.from_numpy(dct_matrix).to(torch_device)

########################################
# compute RD on data with a mean of zero
temp = pickle.load(open('/media/expansion1/spencerkent/Datasets/Kodak/kodak_patches_1mil_8x8_centerd_on_zero.p', 'rb'))
zero_mean_patches = np.transpose(
    temp['batched_patches'], (0, 2, 1)).reshape(
        (-1, patch_dimensions[0]*patch_dimensions[1])).T
img_patch_comp_means = temp['original_patch_means']
centered_patches_test_im_1 = patches_test_im_1 - img_patch_comp_means[:, None]
centered_patches_test_im_2 = patches_test_im_2 - img_patch_comp_means[:, None]

print('Computing DCT codes on centered training data')
centered_patch_codes = invertible_linear.run(
    torch.from_numpy(zero_mean_patches).to(torch_device), torch_dct_matrix,
    orthonormal=True).cpu().numpy()
print('Computing DCT codes on centered test images')
centered_im1_codes = invertible_linear.run(
    torch.from_numpy(centered_patches_test_im_1).to(torch_device),
    torch_dct_matrix, orthonormal=True).cpu().numpy()
centered_im2_codes = invertible_linear.run(
    torch.from_numpy(centered_patches_test_im_2).to(torch_device),
    torch_dct_matrix, orthonormal=True).cpu().numpy()

print('Generating RD curves for centered data')
rates_training_data = []
rates_test_im_1 = []
rates_test_im_2 = []
distortion_training_data = defaultdict(list)
distortion_test_im_1 = defaultdict(list)
distortion_test_im_2 = defaultdict(list)
for rd_idx in range(len(jpeg_quant_multipliers)):
  train_R, train_D, train_cbook, train_huff_ac, train_huff_dc = \
    quantization.jpeg_compute_RD_point(
        centered_patch_codes, zero_mean_patches, dct_matrix,
        quant_multiplier=jpeg_quant_multipliers[rd_idx],
        binwidths=jpeg_hifi_binwidths)

  rates_training_data.append(train_R)
  for metric in train_D:
    distortion_training_data[metric].append(train_D[metric])

  # we can save these to disk to save us time in the future.
  if not os.path.isdir(os.path.abspath(logfile_dir)):
    os.mkdir(logfile_dir)
  if not os.path.isdir(os.path.abspath(logfile_dir + 'huffman_tables')):
    os.mkdir(logfile_dir + 'huffman_tables')
  if not os.path.isdir(os.path.abspath(logfile_dir + 'quant_codebooks')):
    os.mkdir(logfile_dir + 'quant_codebooks')
  pickle.dump(train_huff_ac, open(
    logfile_dir + 'huffman_tables/compression_factor_' + str(rd_idx) +
    '_ac_zeromeanpatches.p', 'wb'))
  pickle.dump(train_huff_dc, open(
    logfile_dir + 'huffman_tables/compression_factor_' + str(rd_idx) +
    '_dc_zeromeanpatches.p', 'wb'))
  pickle.dump(train_cbook, open(
    logfile_dir + 'quant_codebooks/compression_factor_' + str(rd_idx) +
    '_uniformquant_zeromeanpatches.p', 'wb'))

  # now we compute RD points for the test images
  test1_R, test1_D = quantization.jpeg_compute_RD_point(
      centered_im1_codes, centered_patches_test_im_1,
      dct_matrix, precomputed_codebook=train_cbook,
      precomputed_huff_tab_ac=train_huff_ac,
      precomputed_huff_tab_dc=train_huff_dc,
      fullimg_reshape_params={'patch_dim': patch_dimensions,
                              'patch_positions': p_pos_im_1})
  rates_test_im_1.append(test1_R)
  for metric in test1_D:
    distortion_test_im_1[metric].append(test1_D[metric])

  test2_R, test2_D = quantization.jpeg_compute_RD_point(
      centered_im2_codes, centered_patches_test_im_2,
      dct_matrix, precomputed_codebook=train_cbook,
      precomputed_huff_tab_ac=train_huff_ac,
      precomputed_huff_tab_dc=train_huff_dc,
      fullimg_reshape_params={'patch_dim': patch_dimensions,
                              'patch_positions': p_pos_im_2})
  rates_test_im_2.append(test2_R)
  for metric in test2_D:
    distortion_test_im_2[metric].append(test2_D[metric])

rd_performance_to_disk = {'Training data rate': rates_training_data,
                          'Training data distortion': distortion_training_data,
                          'Test img 1 rate': rates_test_im_1,
                          'Test img 1 distortion': distortion_test_im_1,
                          'Test img 2 rate': rates_test_im_2,
                          'Test img 2 distortion': distortion_test_im_2}
pickle.dump(rd_performance_to_disk, open(logfile_dir + 'rd_performance_meanzero_patches.p', 'wb'))

print('Done. Time elapsed: ', time.time() - starttime)


##################################################
# compute RD on data with a constant shift of -128
temp = pickle.load(open('/media/expansion1/spencerkent/Datasets/Kodak/kodak_patches_1mil_8x8_shiftedbyconst128.p', 'rb'))
const_shift_patches = np.transpose(
    temp['batched_patches'], (0, 2, 1)).reshape(
        (-1, patch_dimensions[0]*patch_dimensions[1])).T
const_shift_patches_test_im_1 = patches_test_im_1 - 128
const_shift_patches_test_im_2 = patches_test_im_2 - 128

print('Computing DCT codes on const-shifted training data')
const_shift_patch_codes = invertible_linear.run(
    torch.from_numpy(const_shift_patches).to(torch_device), torch_dct_matrix,
                              orthonormal=True).cpu().numpy()
print('Computing DCT codes on const-shifted test images')
const_shift_im1_codes = invertible_linear.run(
    torch.from_numpy(const_shift_patches_test_im_1).to(torch_device),
    torch_dct_matrix, orthonormal=True).cpu().numpy()
const_shift_im2_codes = invertible_linear.run(
    torch.from_numpy(const_shift_patches_test_im_2).to(torch_device),
    torch_dct_matrix, orthonormal=True).cpu().numpy()

print('Generating RD curves for const-shifted data')
rates_training_data = []
rates_test_im_1 = []
rates_test_im_2 = []
distortion_training_data = defaultdict(list)
distortion_test_im_1 = defaultdict(list)
distortion_test_im_2 = defaultdict(list)
for rd_idx in range(len(jpeg_quant_multipliers)):
  train_R, train_D, train_cbook, train_huff_ac, train_huff_dc = \
    quantization.jpeg_compute_RD_point(
        const_shift_patch_codes, const_shift_patches, dct_matrix,
        quant_multiplier=jpeg_quant_multipliers[rd_idx],
        binwidths=jpeg_hifi_binwidths)

  rates_training_data.append(train_R)
  for metric in train_D:
    distortion_training_data[metric].append(train_D[metric])

  # we can save these to disk to save us time in the future.
  if not os.path.isdir(os.path.abspath(logfile_dir)):
    os.mkdir(logfile_dir)
  if not os.path.isdir(os.path.abspath(logfile_dir + 'huffman_tables')):
    os.mkdir(logfile_dir + 'huffman_tables')
  if not os.path.isdir(os.path.abspath(logfile_dir + 'quant_codebooks')):
    os.mkdir(logfile_dir + 'quant_codebooks')
  pickle.dump(train_huff_ac, open(
    logfile_dir + 'huffman_tables/compression_factor_' + str(rd_idx) +
    '_ac_constshiftpatches.p', 'wb'))
  pickle.dump(train_huff_dc, open(
    logfile_dir + 'huffman_tables/compression_factor_' + str(rd_idx) +
    '_dc_constshiftpatches.p', 'wb'))
  pickle.dump(train_cbook, open(
    logfile_dir + 'quant_codebooks/compression_factor_' + str(rd_idx) +
    '_uniformquant_constshiftpatches.p', 'wb'))

  # now we compute RD points for the test images
  test1_R, test1_D = quantization.jpeg_compute_RD_point(
      const_shift_im1_codes, const_shift_patches_test_im_1,
      dct_matrix, precomputed_codebook=train_cbook,
      precomputed_huff_tab_ac=train_huff_ac,
      precomputed_huff_tab_dc=train_huff_dc,
      fullimg_reshape_params={'patch_dim': patch_dimensions,
                              'patch_positions': p_pos_im_1})
  rates_test_im_1.append(test1_R)
  for metric in test1_D:
    distortion_test_im_1[metric].append(test1_D[metric])

  test2_R, test2_D = quantization.jpeg_compute_RD_point(
      const_shift_im2_codes, const_shift_patches_test_im_2,
      dct_matrix, precomputed_codebook=train_cbook,
      precomputed_huff_tab_ac=train_huff_ac,
      precomputed_huff_tab_dc=train_huff_dc,
      fullimg_reshape_params={'patch_dim': patch_dimensions,
                              'patch_positions': p_pos_im_2})

  rates_test_im_2.append(test2_R)
  for metric in test2_D:
    distortion_test_im_2[metric].append(test2_D[metric])

rd_performance_to_disk = {'Training data rate': rates_training_data,
                          'Training data distortion': distortion_training_data,
                          'Test img 1 rate': rates_test_im_1,
                          'Test img 1 distortion': distortion_test_im_1,
                          'Test img 2 rate': rates_test_im_2,
                          'Test img 2 distortion': distortion_test_im_2}
pickle.dump(rd_performance_to_disk, open(logfile_dir +
  'rd_performance_constshift_patches.p', 'wb'))

print('Done. Time elapsed: ', time.time() - starttime)

print('Generating RD curves with ffmpeg')
# Compute RD curves using ffmpeg
rates_test_im_1, distortion_test_im_1 = quantization.ffmpeg_compute_RD_curve(
    ['/media/expansion1/spencerkent/Datasets/Kodak/raw_images_cropped/kodim08.png'])
rates_test_im_2, distortion_test_im_2 = quantization.ffmpeg_compute_RD_curve(
    ['/media/expansion1/spencerkent/Datasets/Kodak/raw_images_cropped/kodim15.png'])
rd_performance_to_disk = {'Test img 1 rate': rates_test_im_1,
                          'Test img 1 distortion': distortion_test_im_1,
                          'Test img 2 rate': rates_test_im_2,
                          'Test img 2 distortion': distortion_test_im_2}
pickle.dump(rd_performance_to_disk, open(logfile_dir + 'rd_performance_ffmpeg.p', 'wb'))

print('Done. Time elapsed: ', time.time() - starttime)
