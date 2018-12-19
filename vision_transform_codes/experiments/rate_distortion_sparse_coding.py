"""
Computes rate distortion performance for a few different sparse coding schemes

We can vary the sparsity of each code. We can also vary the type of source
coding. For this we can use JPEG's runlength encoding (with and without)
rearranging the dictionary to make this work better, we can use the index
message source code, or we can quantize jointly and try to use a different
source coding scheme
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

from analysis_transforms import fista
from utils import image_processing
from utils import plotting
from utils import quantization

starttime = time.time()
patch_dimensions = (8, 8)
# baseline_binwidths = [10.0 for x in range(np.product(patch_dimensions))]
# quant_multipliers = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 40, 50]

# specific images to visualize the results with
test_images = pickle.load(open('/media/expansion1/spencerkent/Datasets/Kodak/kodak_full_images_testing.p', 'rb'))
patches_test_im_1, p_pos_im_1 = image_processing.patches_from_single_image(
    test_images[0].astype('float32'), patch_dimensions)
patches_test_im_2, p_pos_im_2 = image_processing.patches_from_single_image(
    test_images[1].astype('float32'), patch_dimensions)

torch_device = torch.device('cuda:1')
torch.cuda.set_device(1)

temp = pickle.load(open('/media/expansion1/spencerkent/Datasets/Kodak/kodak_patches_1mil_8x8_centerd_on_zero.p', 'rb'))
zero_mean_patches = np.transpose(
    temp['batched_patches'], (0, 2, 1)).reshape(
        (-1, patch_dimensions[0]*patch_dimensions[1])).T
img_patch_comp_means = temp['original_patch_means']
centered_patches_test_im_1 = patches_test_im_1 - img_patch_comp_means[:, None]
centered_patches_test_im_2 = patches_test_im_2 - img_patch_comp_means[:, None]


# ####################################
# # BASELINE Sparse coding with medium sparsity
# sp = 5.0
# logfile_dir = '/media/expansion1/spencerkent/logfiles/EE290T_project/experiment_logs/SC_medium_sparsity/'
# sc_dictionary = np.load(open('/media/expansion1/spencerkent/logfiles/EE290T_project/experiment_logs/SC_medium_sparsity/trained_dictionary.npy', 'rb'))
#
# print('Computing medium-sparsity sparse codes on centered training data')
# centered_patch_codes = fista.run(
#     torch.from_numpy(zero_mean_patches).to(torch_device),
#     torch.from_numpy(sc_dictionary).to(torch_device), sp, 1000).cpu().numpy()
#
# print('Computing medium-sparsity sparse codes on centered test images')
# centered_im1_codes = fista.run(
#     torch.from_numpy(centered_patches_test_im_1).to(torch_device),
#     torch.from_numpy(sc_dictionary).to(torch_device), sp, 1000).cpu().numpy()
# centered_im2_codes = fista.run(
#     torch.from_numpy(centered_patches_test_im_2).to(torch_device),
#     torch.from_numpy(sc_dictionary).to(torch_device), sp, 1000).cpu().numpy()
#
# print('Generating RD curves for medium-sparsity codes')
# rates_training_data = []
# rates_test_im_1 = []
# rates_test_im_2 = []
# distortion_training_data = defaultdict(list)
# distortion_test_im_1 = defaultdict(list)
# distortion_test_im_2 = defaultdict(list)
# for rd_idx in range(len(quant_multipliers)):
#   train_R, train_D, train_cbook, train_huff_1, train_huff_2 = \
#     quantization.baseline_compute_RD_point(
#         centered_patch_codes, zero_mean_patches, sc_dictionary,
#         quant_multiplier=quant_multipliers[rd_idx],
#         binwidths=baseline_binwidths)
#
#   rates_training_data.append(train_R)
#   for metric in train_D:
#     distortion_training_data[metric].append(train_D[metric])
#
#   # we can save these to disk to save us time in the future.
#   if not os.path.isdir(os.path.abspath(logfile_dir)):
#     os.mkdir(logfile_dir)
#   if not os.path.isdir(os.path.abspath(logfile_dir + 'huffman_tables')):
#     os.mkdir(logfile_dir + 'huffman_tables')
#   if not os.path.isdir(os.path.abspath(logfile_dir + 'quant_codebooks')):
#     os.mkdir(logfile_dir + 'quant_codebooks')
#   pickle.dump(train_huff_1, open(
#     logfile_dir + 'huffman_tables/compression_factor_' + str(rd_idx) +
#     '_tab1.p', 'wb'))
#   pickle.dump(train_huff_2, open(
#     logfile_dir + 'huffman_tables/compression_factor_' + str(rd_idx) +
#     '_tab2.p', 'wb'))
#   pickle.dump(train_cbook, open(
#     logfile_dir + 'quant_codebooks/compression_factor_' + str(rd_idx) +
#     '_uniformquant.p', 'wb'))
#
#   # now we compute RD points for the test images
#   test1_R, test1_D = quantization.baseline_compute_RD_point(
#       centered_im1_codes, centered_patches_test_im_1,
#       sc_dictionary, precomputed_codebook=train_cbook,
#       precomputed_huff_tab1=train_huff_1,
#       precomputed_huff_tab2=train_huff_2,
#       fullimg_reshape_params={'patch_dim': patch_dimensions,
#                               'patch_positions': p_pos_im_1})
#   rates_test_im_1.append(test1_R)
#   for metric in test1_D:
#     distortion_test_im_1[metric].append(test1_D[metric])
#
#   test2_R, test2_D = quantization.baseline_compute_RD_point(
#       centered_im2_codes, centered_patches_test_im_2,
#       sc_dictionary, precomputed_codebook=train_cbook,
#       precomputed_huff_tab1=train_huff_1,
#       precomputed_huff_tab2=train_huff_2,
#       fullimg_reshape_params={'patch_dim': patch_dimensions,
#                               'patch_positions': p_pos_im_2})
#   rates_test_im_2.append(test2_R)
#   for metric in test2_D:
#     distortion_test_im_2[metric].append(test2_D[metric])
#
# rd_performance_to_disk = {'Training data rate': rates_training_data,
#                           'Training data distortion': distortion_training_data,
#                           'Test img 1 rate': rates_test_im_1,
#                           'Test img 1 distortion': distortion_test_im_1,
#                           'Test img 2 rate': rates_test_im_2,
#                           'Test img 2 distortion': distortion_test_im_2}
# pickle.dump(rd_performance_to_disk, open(logfile_dir + 'rd_performance.p', 'wb'))
#
# print('Generating RD curves for medium-sparsity codes, JPEG source coding')
# rates_training_data = []
# rates_test_im_1 = []
# rates_test_im_2 = []
# distortion_training_data = defaultdict(list)
# distortion_test_im_1 = defaultdict(list)
# distortion_test_im_2 = defaultdict(list)
# for rd_idx in range(len(quant_multipliers)):
#   train_R, train_D, train_cbook, train_huff_ac, train_huff_dc = \
#     quantization.jpeg_compute_RD_point(
#         centered_patch_codes, zero_mean_patches, sc_dictionary,
#         quant_multiplier=quant_multipliers[rd_idx],
#         binwidths=baseline_binwidths)
#
#   rates_training_data.append(train_R)
#   for metric in train_D:
#     distortion_training_data[metric].append(train_D[metric])
#
#   # we can save these to disk to save us time in the future.
#   if not os.path.isdir(os.path.abspath(logfile_dir)):
#     os.mkdir(logfile_dir)
#   if not os.path.isdir(os.path.abspath(logfile_dir + 'huffman_tables')):
#     os.mkdir(logfile_dir + 'huffman_tables')
#   if not os.path.isdir(os.path.abspath(logfile_dir + 'quant_codebooks')):
#     os.mkdir(logfile_dir + 'quant_codebooks')
#   pickle.dump(train_huff_ac, open(
#     logfile_dir + 'huffman_tables/compression_factor_' + str(rd_idx) +
#     '_tab1_runlengthcode.p', 'wb'))
#   pickle.dump(train_huff_dc, open(
#     logfile_dir + 'huffman_tables/compression_factor_' + str(rd_idx) +
#     '_tab2_runlengthcode.p', 'wb'))
#
#   # now we compute RD points for the test images
#   test1_R, test1_D = quantization.jpeg_compute_RD_point(
#       centered_im1_codes, centered_patches_test_im_1,
#       sc_dictionary, precomputed_codebook=train_cbook,
#       precomputed_huff_tab_ac=train_huff_ac,
#       precomputed_huff_tab_dc=train_huff_dc,
#       fullimg_reshape_params={'patch_dim': patch_dimensions,
#                               'patch_positions': p_pos_im_1})
#
#   rates_test_im_1.append(test1_R)
#   for metric in test1_D:
#     distortion_test_im_1[metric].append(test1_D[metric])
#
#   test2_R, test2_D = quantization.jpeg_compute_RD_point(
#       centered_im2_codes, centered_patches_test_im_2,
#       sc_dictionary, precomputed_codebook=train_cbook,
#       precomputed_huff_tab_ac=train_huff_ac,
#       precomputed_huff_tab_dc=train_huff_dc,
#       fullimg_reshape_params={'patch_dim': patch_dimensions,
#                               'patch_positions': p_pos_im_2})
#
#   rates_test_im_2.append(test2_R)
#   for metric in test2_D:
#     distortion_test_im_2[metric].append(test2_D[metric])
#
# rd_performance_to_disk = {'Training data rate': rates_training_data,
#                           'Training data distortion': distortion_training_data,
#                           'Test img 1 rate': rates_test_im_1,
#                           'Test img 1 distortion': distortion_test_im_1,
#                           'Test img 2 rate': rates_test_im_2,
#                           'Test img 2 distortion': distortion_test_im_2}
# pickle.dump(rd_performance_to_disk, open(logfile_dir + 'rd_performance_runlengthcode.p', 'wb'))
# print('Done. Time elapsed: ', time.time() - starttime)
#
# ####################################
# #BASELINE Sparse coding with low sparsity
# sp = 1.0
# logfile_dir = '/media/expansion1/spencerkent/logfiles/EE290T_project/experiment_logs/SC_low_sparsity/'
# sc_dictionary = np.load(open('/media/expansion1/spencerkent/logfiles/EE290T_project/experiment_logs/SC_low_sparsity/trained_dictionary.npy', 'rb'))
#
# print('Computing low-sparsity sparse codes on centered training data')
# centered_patch_codes = fista.run(
#     torch.from_numpy(zero_mean_patches).to(torch_device),
#     torch.from_numpy(sc_dictionary).to(torch_device), sp, 1000).cpu().numpy()
#
# print('Computing low-sparsity sparse codes on centered test images')
# centered_im1_codes = fista.run(
#     torch.from_numpy(centered_patches_test_im_1).to(torch_device),
#     torch.from_numpy(sc_dictionary).to(torch_device), sp, 1000).cpu().numpy()
# centered_im2_codes = fista.run(
#     torch.from_numpy(centered_patches_test_im_2).to(torch_device),
#     torch.from_numpy(sc_dictionary).to(torch_device), sp, 1000).cpu().numpy()
#
# print('Generating RD curves for low-sparsity codes')
# rates_training_data = []
# rates_test_im_1 = []
# rates_test_im_2 = []
# distortion_training_data = defaultdict(list)
# distortion_test_im_1 = defaultdict(list)
# distortion_test_im_2 = defaultdict(list)
# for rd_idx in range(len(quant_multipliers)):
#   train_R, train_D, train_cbook, train_huff_1, train_huff_2 = \
#     quantization.baseline_compute_RD_point(
#         centered_patch_codes, zero_mean_patches, sc_dictionary,
#         quant_multiplier=quant_multipliers[rd_idx],
#         binwidths=baseline_binwidths)
#
#   rates_training_data.append(train_R)
#   for metric in train_D:
#     distortion_training_data[metric].append(train_D[metric])
#
#   # we can save these to disk to save us time in the future.
#   if not os.path.isdir(os.path.abspath(logfile_dir)):
#     os.mkdir(logfile_dir)
#   if not os.path.isdir(os.path.abspath(logfile_dir + 'huffman_tables')):
#     os.mkdir(logfile_dir + 'huffman_tables')
#   if not os.path.isdir(os.path.abspath(logfile_dir + 'quant_codebooks')):
#     os.mkdir(logfile_dir + 'quant_codebooks')
#   pickle.dump(train_huff_1, open(
#     logfile_dir + 'huffman_tables/compression_factor_' + str(rd_idx) +
#     '_tab1.p', 'wb'))
#   pickle.dump(train_huff_2, open(
#     logfile_dir + 'huffman_tables/compression_factor_' + str(rd_idx) +
#     '_tab2.p', 'wb'))
#   pickle.dump(train_cbook, open(
#     logfile_dir + 'quant_codebooks/compression_factor_' + str(rd_idx) +
#     '_uniformquant.p', 'wb'))
#
#   # now we compute RD points for the test images
#   test1_R, test1_D = quantization.baseline_compute_RD_point(
#       centered_im1_codes, centered_patches_test_im_1,
#       sc_dictionary, precomputed_codebook=train_cbook,
#       precomputed_huff_tab1=train_huff_1,
#       precomputed_huff_tab2=train_huff_2,
#       fullimg_reshape_params={'patch_dim': patch_dimensions,
#                               'patch_positions': p_pos_im_1})
#   rates_test_im_1.append(test1_R)
#   for metric in test1_D:
#     distortion_test_im_1[metric].append(test1_D[metric])
#
#   test2_R, test2_D = quantization.baseline_compute_RD_point(
#       centered_im2_codes, centered_patches_test_im_2,
#       sc_dictionary, precomputed_codebook=train_cbook,
#       precomputed_huff_tab1=train_huff_1,
#       precomputed_huff_tab2=train_huff_2,
#       fullimg_reshape_params={'patch_dim': patch_dimensions,
#                               'patch_positions': p_pos_im_2})
#   rates_test_im_2.append(test2_R)
#   for metric in test2_D:
#     distortion_test_im_2[metric].append(test2_D[metric])
#
# rd_performance_to_disk = {'Training data rate': rates_training_data,
#                           'Training data distortion': distortion_training_data,
#                           'Test img 1 rate': rates_test_im_1,
#                           'Test img 1 distortion': distortion_test_im_1,
#                           'Test img 2 rate': rates_test_im_2,
#                           'Test img 2 distortion': distortion_test_im_2}
# pickle.dump(rd_performance_to_disk, open(logfile_dir + 'rd_performance.p', 'wb'))
#
# print('Generating RD curves for low-sparsity codes, JPEG source coding')
# rates_training_data = []
# rates_test_im_1 = []
# rates_test_im_2 = []
# distortion_training_data = defaultdict(list)
# distortion_test_im_1 = defaultdict(list)
# distortion_test_im_2 = defaultdict(list)
# for rd_idx in range(len(quant_multipliers)):
#   train_R, train_D, train_cbook, train_huff_ac, train_huff_dc = \
#     quantization.jpeg_compute_RD_point(
#         centered_patch_codes, zero_mean_patches, sc_dictionary,
#         quant_multiplier=quant_multipliers[rd_idx],
#         binwidths=baseline_binwidths)
#
#   rates_training_data.append(train_R)
#   for metric in train_D:
#     distortion_training_data[metric].append(train_D[metric])
#
#   # we can save these to disk to save us time in the future.
#   if not os.path.isdir(os.path.abspath(logfile_dir)):
#     os.mkdir(logfile_dir)
#   if not os.path.isdir(os.path.abspath(logfile_dir + 'huffman_tables')):
#     os.mkdir(logfile_dir + 'huffman_tables')
#   if not os.path.isdir(os.path.abspath(logfile_dir + 'quant_codebooks')):
#     os.mkdir(logfile_dir + 'quant_codebooks')
#   pickle.dump(train_huff_ac, open(
#     logfile_dir + 'huffman_tables/compression_factor_' + str(rd_idx) +
#     '_tab1_runlengthcode.p', 'wb'))
#   pickle.dump(train_huff_dc, open(
#     logfile_dir + 'huffman_tables/compression_factor_' + str(rd_idx) +
#     '_tab2_runlengthcode.p', 'wb'))
#
#   # now we compute RD points for the test images
#   test1_R, test1_D = quantization.jpeg_compute_RD_point(
#       centered_im1_codes, centered_patches_test_im_1,
#       sc_dictionary, precomputed_codebook=train_cbook,
#       precomputed_huff_tab_ac=train_huff_ac,
#       precomputed_huff_tab_dc=train_huff_dc,
#       fullimg_reshape_params={'patch_dim': patch_dimensions,
#                               'patch_positions': p_pos_im_1})
#
#   rates_test_im_1.append(test1_R)
#   for metric in test1_D:
#     distortion_test_im_1[metric].append(test1_D[metric])
#
#   test2_R, test2_D = quantization.jpeg_compute_RD_point(
#       centered_im2_codes, centered_patches_test_im_2,
#       sc_dictionary, precomputed_codebook=train_cbook,
#       precomputed_huff_tab_ac=train_huff_ac,
#       precomputed_huff_tab_dc=train_huff_dc,
#       fullimg_reshape_params={'patch_dim': patch_dimensions,
#                               'patch_positions': p_pos_im_2})
#
#   rates_test_im_2.append(test2_R)
#   for metric in test2_D:
#     distortion_test_im_2[metric].append(test2_D[metric])
#
# rd_performance_to_disk = {'Training data rate': rates_training_data,
#                           'Training data distortion': distortion_training_data,
#                           'Test img 1 rate': rates_test_im_1,
#                           'Test img 1 distortion': distortion_test_im_1,
#                           'Test img 2 rate': rates_test_im_2,
#                           'Test img 2 distortion': distortion_test_im_2}
# pickle.dump(rd_performance_to_disk, open(logfile_dir + 'rd_performance_runlengthcode.p', 'wb'))
# print('Done. Time elapsed: ', time.time() - starttime)
#
# ####################################
# #BASELINE Sparse coding with high sparsity
# sp = 10.0
# logfile_dir = '/media/expansion1/spencerkent/logfiles/EE290T_project/experiment_logs/SC_high_sparsity/'
# sc_dictionary = np.load(open('/media/expansion1/spencerkent/logfiles/EE290T_project/experiment_logs/SC_high_sparsity/trained_dictionary.npy', 'rb'))
#
# print('Computing high-sparsity sparse codes on centered training data')
# centered_patch_codes = fista.run(
#     torch.from_numpy(zero_mean_patches).to(torch_device),
#     torch.from_numpy(sc_dictionary).to(torch_device), sp, 1000).cpu().numpy()
#
# print('Computing high-sparsity sparse codes on centered test images')
# centered_im1_codes = fista.run(
#     torch.from_numpy(centered_patches_test_im_1).to(torch_device),
#     torch.from_numpy(sc_dictionary).to(torch_device), sp, 1000).cpu().numpy()
# centered_im2_codes = fista.run(
#     torch.from_numpy(centered_patches_test_im_2).to(torch_device),
#     torch.from_numpy(sc_dictionary).to(torch_device), sp, 1000).cpu().numpy()
#
# print('Generating RD curves for high-sparsity codes')
# rates_training_data = []
# rates_test_im_1 = []
# rates_test_im_2 = []
# distortion_training_data = defaultdict(list)
# distortion_test_im_1 = defaultdict(list)
# distortion_test_im_2 = defaultdict(list)
# for rd_idx in range(len(quant_multipliers)):
#   train_R, train_D, train_cbook, train_huff_1, train_huff_2 = \
#     quantization.baseline_compute_RD_point(
#         centered_patch_codes, zero_mean_patches, sc_dictionary,
#         quant_multiplier=quant_multipliers[rd_idx],
#         binwidths=baseline_binwidths)
#
#   rates_training_data.append(train_R)
#   for metric in train_D:
#     distortion_training_data[metric].append(train_D[metric])
#
#   # we can save these to disk to save us time in the future.
#   if not os.path.isdir(os.path.abspath(logfile_dir)):
#     os.mkdir(logfile_dir)
#   if not os.path.isdir(os.path.abspath(logfile_dir + 'huffman_tables')):
#     os.mkdir(logfile_dir + 'huffman_tables')
#   if not os.path.isdir(os.path.abspath(logfile_dir + 'quant_codebooks')):
#     os.mkdir(logfile_dir + 'quant_codebooks')
#   pickle.dump(train_huff_1, open(
#     logfile_dir + 'huffman_tables/compression_factor_' + str(rd_idx) +
#     '_tab1.p', 'wb'))
#   pickle.dump(train_huff_2, open(
#     logfile_dir + 'huffman_tables/compression_factor_' + str(rd_idx) +
#     '_tab2.p', 'wb'))
#   pickle.dump(train_cbook, open(
#     logfile_dir + 'quant_codebooks/compression_factor_' + str(rd_idx) +
#     '_uniformquant.p', 'wb'))
#
#   # now we compute RD points for the test images
#   test1_R, test1_D = quantization.baseline_compute_RD_point(
#       centered_im1_codes, centered_patches_test_im_1,
#       sc_dictionary, precomputed_codebook=train_cbook,
#       precomputed_huff_tab1=train_huff_1,
#       precomputed_huff_tab2=train_huff_2,
#       fullimg_reshape_params={'patch_dim': patch_dimensions,
#                               'patch_positions': p_pos_im_1})
#   rates_test_im_1.append(test1_R)
#   for metric in test1_D:
#     distortion_test_im_1[metric].append(test1_D[metric])
#
#   test2_R, test2_D = quantization.baseline_compute_RD_point(
#       centered_im2_codes, centered_patches_test_im_2,
#       sc_dictionary, precomputed_codebook=train_cbook,
#       precomputed_huff_tab1=train_huff_1,
#       precomputed_huff_tab2=train_huff_2,
#       fullimg_reshape_params={'patch_dim': patch_dimensions,
#                               'patch_positions': p_pos_im_2})
#   rates_test_im_2.append(test2_R)
#   for metric in test2_D:
#     distortion_test_im_2[metric].append(test2_D[metric])
#
# rd_performance_to_disk = {'Training data rate': rates_training_data,
#                           'Training data distortion': distortion_training_data,
#                           'Test img 1 rate': rates_test_im_1,
#                           'Test img 1 distortion': distortion_test_im_1,
#                           'Test img 2 rate': rates_test_im_2,
#                           'Test img 2 distortion': distortion_test_im_2}
# pickle.dump(rd_performance_to_disk, open(logfile_dir + 'rd_performance.p', 'wb'))
#
# print('Generating RD curves for high-sparsity codes, JPEG source coding')
# rates_training_data = []
# rates_test_im_1 = []
# rates_test_im_2 = []
# distortion_training_data = defaultdict(list)
# distortion_test_im_1 = defaultdict(list)
# distortion_test_im_2 = defaultdict(list)
# for rd_idx in range(len(quant_multipliers)):
#   train_R, train_D, train_cbook, train_huff_ac, train_huff_dc = \
#     quantization.jpeg_compute_RD_point(
#         centered_patch_codes, zero_mean_patches, sc_dictionary,
#         quant_multiplier=quant_multipliers[rd_idx],
#         binwidths=baseline_binwidths)
#
#   rates_training_data.append(train_R)
#   for metric in train_D:
#     distortion_training_data[metric].append(train_D[metric])
#
#   # we can save these to disk to save us time in the future.
#   if not os.path.isdir(os.path.abspath(logfile_dir)):
#     os.mkdir(logfile_dir)
#   if not os.path.isdir(os.path.abspath(logfile_dir + 'huffman_tables')):
#     os.mkdir(logfile_dir + 'huffman_tables')
#   if not os.path.isdir(os.path.abspath(logfile_dir + 'quant_codebooks')):
#     os.mkdir(logfile_dir + 'quant_codebooks')
#   pickle.dump(train_huff_ac, open(
#     logfile_dir + 'huffman_tables/compression_factor_' + str(rd_idx) +
#     '_tab1_runlengthcode.p', 'wb'))
#   pickle.dump(train_huff_dc, open(
#     logfile_dir + 'huffman_tables/compression_factor_' + str(rd_idx) +
#     '_tab2_runlengthcode.p', 'wb'))
#
#   # now we compute RD points for the test images
#   test1_R, test1_D = quantization.jpeg_compute_RD_point(
#       centered_im1_codes, centered_patches_test_im_1,
#       sc_dictionary, precomputed_codebook=train_cbook,
#       precomputed_huff_tab_ac=train_huff_ac,
#       precomputed_huff_tab_dc=train_huff_dc,
#       fullimg_reshape_params={'patch_dim': patch_dimensions,
#                               'patch_positions': p_pos_im_1})
#
#   rates_test_im_1.append(test1_R)
#   for metric in test1_D:
#     distortion_test_im_1[metric].append(test1_D[metric])
#
#   test2_R, test2_D = quantization.jpeg_compute_RD_point(
#       centered_im2_codes, centered_patches_test_im_2,
#       sc_dictionary, precomputed_codebook=train_cbook,
#       precomputed_huff_tab_ac=train_huff_ac,
#       precomputed_huff_tab_dc=train_huff_dc,
#       fullimg_reshape_params={'patch_dim': patch_dimensions,
#                               'patch_positions': p_pos_im_2})
#
#   rates_test_im_2.append(test2_R)
#   for metric in test2_D:
#     distortion_test_im_2[metric].append(test2_D[metric])
#
# rd_performance_to_disk = {'Training data rate': rates_training_data,
#                           'Training data distortion': distortion_training_data,
#                           'Test img 1 rate': rates_test_im_1,
#                           'Test img 1 distortion': distortion_test_im_1,
#                           'Test img 2 rate': rates_test_im_2,
#                           'Test img 2 distortion': distortion_test_im_2}
# pickle.dump(rd_performance_to_disk, open(logfile_dir + 'rd_performance_runlengthcode.p', 'wb'))
# print('Done. Time elapsed: ', time.time() - starttime)


# ####################################
# # MOD1 Sparse coding with medium sparsity
# # reset quant multiplier and baseline binwidth for Lloyd
# baseline_binwidths = [5.0 for x in range(np.product(patch_dimensions))]
# quant_multipliers = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
#
# sp = 5.0
# logfile_dir = '/media/expansion1/spencerkent/logfiles/EE290T_project/experiment_logs/SC_medium_sparsity/'
# sc_dictionary = np.load(open('/media/expansion1/spencerkent/logfiles/EE290T_project/experiment_logs/SC_medium_sparsity/trained_dictionary.npy', 'rb'))
#
# print('Computing medium-sparsity sparse codes on centered training data')
# centered_patch_codes = fista.run(
#     torch.from_numpy(zero_mean_patches).to(torch_device),
#     torch.from_numpy(sc_dictionary).to(torch_device), sp, 1000).cpu().numpy()
#
# print('Computing medium-sparsity sparse codes on centered test images')
# centered_im1_codes = fista.run(
#     torch.from_numpy(centered_patches_test_im_1).to(torch_device),
#     torch.from_numpy(sc_dictionary).to(torch_device), sp, 1000).cpu().numpy()
# centered_im2_codes = fista.run(
#     torch.from_numpy(centered_patches_test_im_2).to(torch_device),
#     torch.from_numpy(sc_dictionary).to(torch_device), sp, 1000).cpu().numpy()
#
#
# print('Generating RD curves for MOD1 medium-sparsity codes')
# rates_training_data = []
# rates_test_im_1 = []
# rates_test_im_2 = []
# distortion_training_data = defaultdict(list)
# distortion_test_im_1 = defaultdict(list)
# distortion_test_im_2 = defaultdict(list)
# for rd_idx in range(len(quant_multipliers)):
#   train_R, train_D, train_cbook, train_cw_len, train_huff_1 = \
#     quantization.Mod1_compute_RD_point(
#         centered_patch_codes[:, 0:100000], zero_mean_patches[:, 0:100000], 
#         sc_dictionary, quant_multiplier=quant_multipliers[rd_idx],
#         init_binwidths=baseline_binwidths)
#
#   rates_training_data.append(train_R)
#   for metric in train_D:
#     distortion_training_data[metric].append(train_D[metric])
#
#   # we can save these to disk to save us time in the future.
#   if not os.path.isdir(os.path.abspath(logfile_dir)):
#     os.mkdir(logfile_dir)
#   if not os.path.isdir(os.path.abspath(logfile_dir + 'huffman_tables')):
#     os.mkdir(logfile_dir + 'huffman_tables')
#   if not os.path.isdir(os.path.abspath(logfile_dir + 'quant_codebooks')):
#     os.mkdir(logfile_dir + 'quant_codebooks')
#   pickle.dump(train_huff_1, open(
#     logfile_dir + 'huffman_tables/Mod1_SIMPLEST_compression_factor_' + str(rd_idx) +
#     '_tab1.p', 'wb'))
#   # pickle.dump(train_huff_2, open(
#   #   logfile_dir + 'huffman_tables/Mod1_SIMPLE_compression_factor_' + str(rd_idx) +
#   #   '_tab2.p', 'wb'))
#   pickle.dump(train_cbook, open(
#     logfile_dir + 'quant_codebooks/Mod1_SIMPLEST_compression_factor_' + str(rd_idx) +
#     '_scalarlloyd.p', 'wb'))
#
#   # now we compute RD points for the test images
#   test1_R, test1_D = quantization.Mod1_compute_RD_point(
#       centered_im1_codes, centered_patches_test_im_1,
#       sc_dictionary, quant_multiplier=quant_multipliers[rd_idx],
#       precomputed_codebook=train_cbook,
#       precomputed_codebook_lengths=train_cw_len,
#       precomputed_huff_tab1=train_huff_1,
#       # precomputed_huff_tab2=train_huff_2,
#       fullimg_reshape_params={'patch_dim': patch_dimensions,
#                               'patch_positions': p_pos_im_1})
#   rates_test_im_1.append(test1_R)
#   for metric in test1_D:
#     distortion_test_im_1[metric].append(test1_D[metric])
#
#   test2_R, test2_D = quantization.Mod1_compute_RD_point(
#       centered_im2_codes, centered_patches_test_im_2,
#       sc_dictionary, quant_multiplier=quant_multipliers[rd_idx],
#       precomputed_codebook=train_cbook,
#       precomputed_codebook_lengths=train_cw_len,
#       precomputed_huff_tab1=train_huff_1,
#       # precomputed_huff_tab2=train_huff_2,
#       fullimg_reshape_params={'patch_dim': patch_dimensions,
#                               'patch_positions': p_pos_im_2})
#   rates_test_im_2.append(test2_R)
#   for metric in test2_D:
#     distortion_test_im_2[metric].append(test2_D[metric])
#
# rd_performance_to_disk = {'Training data rate': rates_training_data,
#                           'Training data distortion': distortion_training_data,
#                           'Test img 1 rate': rates_test_im_1,
#                           'Test img 1 distortion': distortion_test_im_1,
#                           'Test img 2 rate': rates_test_im_2,
#                           'Test img 2 distortion': distortion_test_im_2}
# pickle.dump(rd_performance_to_disk, open(logfile_dir + 'Mod1_SIMPLEST_rd_performance.p', 'wb'))

# ####################################
# # MOD2 Sparse coding with medium sparsity
# coeff_subsets_SW = [[15], [1], [27], [44], [20], [25], [37], [63], [2], [21], 
#                     [16], [42], [10], [40], [50], [55], [34], [62], [35], [51],
#                     [58], [47], [9], [52], [11], [14], [46], [49], [13], [26], 
#                     [5], [60], [61], [8], [3], [7], [57], [12], [6], [54], [41],
#                     [0, 4, 17, 18, 19, 22, 23, 24, 28, 29, 30, 31, 32, 33, 36,
#                      38, 39, 43, 45, 48, 53, 56, 59]]
# scal_clusts = [x[0] for x in coeff_subsets_SW[:-1]]
# vec_clust = coeff_subsets_SW[-1]
#
# scal_baseline_binwidths = [5.0 for x in range(len(scal_clusts))]
# scal_quant_multipliers = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30]
# vec_max_bin_init = 100000
# vec_quant_multipliers = [0.25, 0.5, 1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 11.0, 13.0]
# assert len(scal_quant_multipliers) == len(vec_quant_multipliers)
#
# sp = 5.0
# logfile_dir = '/media/expansion1/spencerkent/logfiles/EE290T_project/experiment_logs/SC_medium_sparsity/'
# sc_dictionary = np.load(open('/media/expansion1/spencerkent/logfiles/EE290T_project/experiment_logs/SC_medium_sparsity/trained_dictionary.npy', 'rb'))
#
# print('Computing medium-sparsity sparse codes on centered training data')
# centered_patch_codes = fista.run(
#     torch.from_numpy(zero_mean_patches).to(torch_device),
#     torch.from_numpy(sc_dictionary).to(torch_device), sp, 1000).cpu().numpy()
#
# print('Computing medium-sparsity sparse codes on centered test images')
# centered_im1_codes = fista.run(
#     torch.from_numpy(centered_patches_test_im_1).to(torch_device),
#     torch.from_numpy(sc_dictionary).to(torch_device), sp, 1000).cpu().numpy()
# centered_im2_codes = fista.run(
#     torch.from_numpy(centered_patches_test_im_2).to(torch_device),
#     torch.from_numpy(sc_dictionary).to(torch_device), sp, 1000).cpu().numpy()
#
#
# print('Generating RD curves HEY for MOD2 medium-sparsity codes')
# rates_training_data = []
# rates_test_im_1 = []
# rates_test_im_2 = []
# distortion_training_data = defaultdict(list)
# distortion_test_im_1 = defaultdict(list)
# distortion_test_im_2 = defaultdict(list)
# for rd_idx in range(len(scal_quant_multipliers)):
#   train_R, train_D, train_scal_cbook, train_vec_cbook, train_vec_cw_len, train_huff_1, train_huff_2, train_huff_3 = quantization.Mod2_compute_RD_point(
#         centered_patch_codes[:, 0:100000], zero_mean_patches[:, 0:100000], 
#         sc_dictionary, scal_clusts, vec_clust, 
#         scal_quant_multiplier=scal_quant_multipliers[rd_idx],
#         scal_binwidths=scal_baseline_binwidths, 
#         vec_quant_multiplier=vec_quant_multipliers[rd_idx],
#         vec_init_num_bins=vec_max_bin_init)
#
#   rates_training_data.append(train_R)
#   for metric in train_D:
#     distortion_training_data[metric].append(train_D[metric])
#
#   # we can save these to disk to save us time in the future.
#   if not os.path.isdir(os.path.abspath(logfile_dir)):
#     os.mkdir(logfile_dir)
#   if not os.path.isdir(os.path.abspath(logfile_dir + 'huffman_tables')):
#     os.mkdir(logfile_dir + 'huffman_tables')
#   if not os.path.isdir(os.path.abspath(logfile_dir + 'quant_codebooks')):
#     os.mkdir(logfile_dir + 'quant_codebooks')
#   pickle.dump(train_huff_1, open(
#     logfile_dir + 'huffman_tables/Mod2_compression_factor_' + str(rd_idx) +
#     '_tab1.p', 'wb'))
#   pickle.dump(train_huff_2, open(
#     logfile_dir + 'huffman_tables/Mod2_compression_factor_' + str(rd_idx) +
#     '_tab2.p', 'wb'))
#   pickle.dump(train_huff_3, open(
#     logfile_dir + 'huffman_tables/Mod2_compression_factor_' + str(rd_idx) +
#     '_tab3.p', 'wb'))
#   pickle.dump(train_scal_cbook, open(
#     logfile_dir + 'quant_codebooks/Mod2_compression_factor_' + str(rd_idx) +
#     '_scalcodebook.p', 'wb'))
#   pickle.dump(train_vec_cbook, open(
#     logfile_dir + 'quant_codebooks/Mod2_compression_factor_' + str(rd_idx) +
#     '_veccodebook.p', 'wb'))
#   pickle.dump(train_vec_cw_len, open(
#     logfile_dir + 'quant_codebooks/Mod2_compression_factor_' + str(rd_idx) +
#     '_veccodebook_lengths.p', 'wb'))
#
#   # now we compute RD points for the test images
#   test1_R, test1_D = quantization.Mod2_compute_RD_point(
#       centered_im1_codes, centered_patches_test_im_1,
#       sc_dictionary, scal_clusts, vec_clust, 
#       vec_quant_multiplier=vec_quant_multipliers[rd_idx],
#       precomputed_scal_codebook=train_scal_cbook,
#       precomputed_vec_codebook=train_vec_cbook,
#       precomputed_vec_codebook_lengths=train_vec_cw_len,
#       precomputed_huff_tab1=train_huff_1,
#       precomputed_huff_tab2=train_huff_2,
#       precomputed_huff_tab3=train_huff_3,
#       fullimg_reshape_params={'patch_dim': patch_dimensions,
#                               'patch_positions': p_pos_im_1})
#   rates_test_im_1.append(test1_R)
#   for metric in test1_D:
#     distortion_test_im_1[metric].append(test1_D[metric])
#
#   test2_R, test2_D = quantization.Mod2_compute_RD_point(
#       centered_im2_codes, centered_patches_test_im_2,
#       sc_dictionary, scal_clusts, vec_clust, 
#       vec_quant_multiplier=vec_quant_multipliers[rd_idx],
#       precomputed_scal_codebook=train_scal_cbook,
#       precomputed_vec_codebook=train_vec_cbook,
#       precomputed_vec_codebook_lengths=train_vec_cw_len,
#       precomputed_huff_tab1=train_huff_1,
#       precomputed_huff_tab2=train_huff_2,
#       precomputed_huff_tab3=train_huff_3,
#       fullimg_reshape_params={'patch_dim': patch_dimensions,
#                               'patch_positions': p_pos_im_2})
#   rates_test_im_2.append(test2_R)
#   for metric in test2_D:
#     distortion_test_im_2[metric].append(test2_D[metric])
#
# rd_performance_to_disk = {'Training data rate': rates_training_data,
#                           'Training data distortion': distortion_training_data,
#                           'Test img 1 rate': rates_test_im_1,
#                           'Test img 1 distortion': distortion_test_im_1,
#                           'Test img 2 rate': rates_test_im_2,
#                           'Test img 2 distortion': distortion_test_im_2}
# pickle.dump(rd_performance_to_disk, open(logfile_dir + 'Mod2_rd_performance.p', 'wb'))

####################################
# MOD3 Sparse coding with medium sparsity
coeff_subsets_SW = [[15], [1], [27], [44], [20], [25], [37], [63], [2], [21], 
                    [16], [42], [10], [40], [50], [55], [34], [62], [35], [51],
                    [58], [47], [9], [52], [11], [14], [46], [49], [13], [26], 
                    [5], [60], [61], [8], [3], [7], [57], [12], [6], [54], [41],
                    [0, 4, 17, 18, 19, 22, 23, 24, 28, 29, 30, 31, 32, 33, 36,
                     38, 39, 43, 45, 48, 53, 56, 59]]
scal_clusts = [x[0] for x in coeff_subsets_SW[:-1]]
vec_clust = coeff_subsets_SW[-1]

scal_baseline_binwidths = [5.0 for x in range(len(scal_clusts))]
scal_quant_multipliers = [1, 2, 4, 10, 20]
vec_max_bin_init = 100000
vec_quant_multipliers = [0.25, 1.0, 4.0, 8.0, 13.0]
assert len(scal_quant_multipliers) == len(vec_quant_multipliers)

sp = 5.0
logfile_dir = '/media/expansion1/spencerkent/logfiles/EE290T_project/experiment_logs/SC_medium_sparsity/'
sc_dictionary = np.load(open('/media/expansion1/spencerkent/logfiles/EE290T_project/experiment_logs/SC_medium_sparsity/trained_dictionary.npy', 'rb'))

print('Computing medium-sparsity sparse codes on centered training data')
centered_patch_codes = fista.run(
    torch.from_numpy(zero_mean_patches).to(torch_device),
    torch.from_numpy(sc_dictionary).to(torch_device), sp, 1000).cpu().numpy()

print('Computing medium-sparsity sparse codes on centered test images')
centered_im1_codes = fista.run(
    torch.from_numpy(centered_patches_test_im_1).to(torch_device),
    torch.from_numpy(sc_dictionary).to(torch_device), sp, 1000).cpu().numpy()
centered_im2_codes = fista.run(
    torch.from_numpy(centered_patches_test_im_2).to(torch_device),
    torch.from_numpy(sc_dictionary).to(torch_device), sp, 1000).cpu().numpy()


print('Generating RD curves HEY for MOD3 medium-sparsity codes')
rates_training_data = []
rates_test_im_1 = []
rates_test_im_2 = []
distortion_training_data = defaultdict(list)
distortion_test_im_1 = defaultdict(list)
distortion_test_im_2 = defaultdict(list)
for rd_idx in range(len(scal_quant_multipliers)):
  train_R, train_D, train_scal_cbook, train_vec_cbook, train_vec_cw_len, train_huff_1, train_huff_2, train_huff_3 = quantization.Mod3_compute_RD_point(
        centered_patch_codes[:, 0:100000], zero_mean_patches[:, 0:100000], 
        sc_dictionary, scal_clusts, vec_clust, 
        scal_quant_multiplier=scal_quant_multipliers[rd_idx],
        scal_binwidths=scal_baseline_binwidths, 
        vec_quant_multiplier=vec_quant_multipliers[rd_idx],
        vec_init_num_bins=vec_max_bin_init)

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
  pickle.dump(train_huff_1, open(
    logfile_dir + 'huffman_tables/Mod3_compression_factor_' + str(rd_idx) +
    '_tab1.p', 'wb'))
  pickle.dump(train_huff_2, open(
    logfile_dir + 'huffman_tables/Mod3_compression_factor_' + str(rd_idx) +
    '_tab2.p', 'wb'))
  pickle.dump(train_huff_3, open(
    logfile_dir + 'huffman_tables/Mod3_compression_factor_' + str(rd_idx) +
    '_tab3.p', 'wb'))
  pickle.dump(train_scal_cbook, open(
    logfile_dir + 'quant_codebooks/Mod3_compression_factor_' + str(rd_idx) +
    '_scalcodebook.p', 'wb'))
  pickle.dump(train_vec_cbook, open(
    logfile_dir + 'quant_codebooks/Mod3_compression_factor_' + str(rd_idx) +
    '_veccodebook.p', 'wb'))
  pickle.dump(train_vec_cw_len, open(
    logfile_dir + 'quant_codebooks/Mod3_compression_factor_' + str(rd_idx) +
    '_veccodebook_lengths.p', 'wb'))

  # now we compute RD points for the test images
  test1_R, test1_D = quantization.Mod3_compute_RD_point(
      centered_im1_codes, centered_patches_test_im_1,
      sc_dictionary, scal_clusts, vec_clust, 
      vec_quant_multiplier=vec_quant_multipliers[rd_idx],
      precomputed_scal_codebook=train_scal_cbook,
      precomputed_vec_codebook=train_vec_cbook,
      precomputed_vec_codebook_lengths=train_vec_cw_len,
      precomputed_huff_tab1=train_huff_1,
      precomputed_huff_tab2=train_huff_2,
      precomputed_huff_tab3=train_huff_3,
      fullimg_reshape_params={'patch_dim': patch_dimensions,
                              'patch_positions': p_pos_im_1})
  rates_test_im_1.append(test1_R)
  for metric in test1_D:
    distortion_test_im_1[metric].append(test1_D[metric])

  test2_R, test2_D = quantization.Mod3_compute_RD_point(
      centered_im2_codes, centered_patches_test_im_2,
      sc_dictionary, scal_clusts, vec_clust, 
      vec_quant_multiplier=vec_quant_multipliers[rd_idx],
      precomputed_scal_codebook=train_scal_cbook,
      precomputed_vec_codebook=train_vec_cbook,
      precomputed_vec_codebook_lengths=train_vec_cw_len,
      precomputed_huff_tab1=train_huff_1,
      precomputed_huff_tab2=train_huff_2,
      precomputed_huff_tab3=train_huff_3,
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
pickle.dump(rd_performance_to_disk, open(logfile_dir + 'Mod3_rd_performance.p', 'wb'))
