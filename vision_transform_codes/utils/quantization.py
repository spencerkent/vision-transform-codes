"""
Just some tools for quantization
"""
import os
import shutil
from collections import defaultdict
import numpy as np
import cv2
from skimage.measure import compare_ssim as ssim

import sys
sys.path.insert(0, '/home/spencerkent/Projects/generalized-lloyd-quantization/')
from generalized_lloyd_quantization import null_uniform as uniform_quant
from generalized_lloyd_quantization import optimal_generalized_lloyd_LBG as lloyd_quant
from generalized_lloyd_quantization import LSTC_optimal_generalized_lloyd_LBG as lstc_lloyd_quant

from . import jpeg
from . import sparse_source_coding
from . import image_processing

#######################
# Quantization variants

# 1. Baseline   (uniform scalar quant, index coding)
# 2. Mod1  (Lloyd scalar quantization on Nonzero coeffs, index coding)
# 3. Mod2  clustered coeffs, uniform scalar quant, index coding of singular
#          cluster, (indexes, '*', values). Vector Lloyd, Then Huffman code 
#          of joint vector
# 4. Mod3  clustered coeffs, uniform scalar quant, index coding of singular
#          cluster, (indexes, '*', values). Transform aware Lloyd, Then Huffman 
#          code of joint vector


def cbook_inds_of_zero_pts(codebook):
  """
  Returns the index of the zero-valued codeword for each dimension

  Parameters
  ----------
  codebook : list(ndarray(float))
      Each element corresponds to the codebook for one dimension of our code.
      The number of codewords will be variable which is why we can't store
      these all in one big array.

  Returns
  -------
  zero_assign_inds : ndarray(int)
      The 1d array giving the indexes, in each dimension's respective codebook,
      of the codeword that has value precisely zero. For any codebook that
      doesn't have any codeword with value zero, we return -1
  """
  assert codebook[0].ndim == 1  # TODO: make this work for vector codewords
  zero_assign_inds = np.zeros((len(codebook),), dtype='int')
  for coeff_idx in range(len(codebook)):
    temp = np.where(codebook[coeff_idx] == 0)[0]
    if len(temp) == 1:
      zero_assign_inds[coeff_idx] = temp[0]
    elif len(temp) == 0:
      # no assignments points were directly at zero. We'll assign the dummy idx
      zero_assign_inds[coeff_idx] = -1
    else:
      raise ValueError('Uh-oh, there were two or more assignment pts at zero')

  return zero_assign_inds


def num_cword_per_dim(codebook):
  assert codebook[0].ndim == 1  # TODO: make this work for vector codewords
  cwords_per_dim = []
  for dim_idx in range(len(codebook)):
    cwords_per_dim.append(len(codebook[dim_idx]))

  return cwords_per_dim


def compute_rMSE(target, reconstruction):
  return np.sqrt(np.mean(np.square(target - reconstruction)))


def compute_pSNR(target, reconstruction):
  signal_magnitude = np.max(target) - np.min(target)
  MSE = np.mean(np.square(target - reconstruction))
  return 10. * np.log10((signal_magnitude**2)/MSE)


def compute_ssim(target, reconstruction):
  # this designed to be run on a pair of n x m images, not really batches...
  signal_magnitude = np.max(target) - np.min(target)
  # these are the settings that the scikit-image documentation indicates
  # match the ones chosen in the original SSIM paper (Wang 2004, I believe).
  return ssim(target, reconstruction, data_range=signal_magnitude,
              gaussian_weights=True, sigma=1.5, use_sample_covariance=False)


def ffmpeg_compute_RD_curve(filepath_list):

  # create a temporary directory to store the compressed files. We'll delete
  # it after we're done here
  os.mkdir('temp_ffmpeg_file_outputs')

  avg_rates = []
  avg_distortions = defaultdict(list)
  for quality_level in range(1, 32):
    rates = []  # bits per pixel
    distortions = defaultdict(list)
    for file_idx in range(len(filepath_list)):
      os.system(
          "ffmpeg -i %s -q:v %s -y temp_ffmpeg_file_outputs/temp.jpg" %
          (filepath_list[file_idx], str(quality_level)))
      ground_truth = cv2.imread(filepath_list[file_idx], cv2.IMREAD_GRAYSCALE)
      compressed_img = cv2.imread('temp_ffmpeg_file_outputs/temp.jpg',
                                  cv2.IMREAD_GRAYSCALE)
      # in our manual JPEG implementation the original pixel values have 128
      # subtracted from them, so to make sure that nothing funky is going
      # on we'll compute distortion on these shifted images
      distortions['pSNR'].append(compute_pSNR(ground_truth - 128,
                                              compressed_img - 128))
      distortions['rMSE'].append(compute_rMSE(ground_truth - 128,
                                              compressed_img - 128))
      distortions['SSIM'].append(compute_ssim(ground_truth - 128,
                                              compressed_img - 128))
      rates.append(os.path.getsize('temp_ffmpeg_file_outputs/temp.jpg') * 8
                   / np.product(ground_truth.shape))

    avg_rates.append(np.mean(rates))
    for metric in distortions:
      avg_distortions[metric].append(np.mean(distortions[metric]))

  shutil.rmtree('temp_ffmpeg_file_outputs')

  return avg_rates, avg_distortions


def jpeg_compute_RD_point(raw_codes, target_patches, dictionary,
                          quant_multiplier=None, binwidths=None,
                          precomputed_codebook=None,
                          precomputed_huff_tab_ac=None,
                          precomputed_huff_tab_dc=None,
                          fullimg_reshape_params=None):
  """
  Computes a rate-distortion pair for JPEG source coding

  When we have yet to set the Huffman tables for the AC and DC components of
  the code, we call this without the last three parameters above. Once this
  has been computed on a training set, we can call this with those parameters
  set in order to (much more quickly) compute an RD point for some data.

  Parameters
  ----------
  raw_codes : ndarray(float) size=(s, D)
      The dataset of D samples of codes of size s
  target_patches : ndarray(float) size=(n, D)
      The D target patches for each of these codes (each has dim n)
  dictionary : ndarray(float) size=(n, s)
      The dictionary matrix that synthesizes image patches. y=Ax, and this is A
  quant_multiplier : float, optional
      This is the 'quality' parameter for jpeg. It multiplies the hifi binwidths
  binwidths : ndarray(float), optional
      These are the baseline binwidths for jpeg that we want to use for each
      coefficient of the code.
  precomputed_codebook : list(ndarray), optional
      The length s list that gives assignment points for each dimension of the
      code. The number of assignment points will be variable per dimension.
  precomputed_huff_tab_ac : dictionary(str), optional
      Just a lookup table that converts JPEG symbol 1 (which is an 8-bit number
      represented by its hex string) into a binary string
  precomputed_huff_tab_dc : dictionary(str), optional
      Just a lookup table that converts JPEG dc 'category' value (just the
      number of bits required to code the actual amplitude), a hex string,
      into a binary string.
  fullimg_reshape_params = dictionary, optional
      If provided, then we reshape the reconstructed patches and the ground
      truth patches into a full image which we can comute the SSIM on. I
      don't believe we should be be computing the SSIM on random patches.

  Returns
  -------
  rate : float
      The number of bits per code coefficient, on average
  distortion : float
      The Peak Signal to Noise Ratio of the reconstructed patches, after
      the codes are quantized.
  codebook : list(ndarray), if precomputed_codebook is None
      The length s list that gives assignment points for each dimension of the
      code. The number of assignment points will be variable per dimension.
  huff_tab_ac : dictionary(str), if precomputed_huff_tab_ac is None
      Just a lookup table that converts JPEG symbol 1 (which is an 8-bit number
      represented by its hex string) into a binary string
  huff_tab_dc : dictionary(str), if precomputed_huff_tab_ac is None
      Just a lookup table that converts JPEG dc 'category' value (just the
      number of bits required to code the actual amplitude), a hex string,
      into a binary string.
  """
  assert raw_codes.shape[0] == 64, 'Havent tested on anything other than 8x8'
  training = True if precomputed_codebook == None else False
  if training:
    codebook = []
  else:
    codebook = precomputed_codebook
    huff_tab_ac = precomputed_huff_tab_ac
    huff_tab_dc = precomputed_huff_tab_dc

  all_assignments = np.zeros(raw_codes.T.shape, dtype='int')
  quantized_codes = np.zeros(raw_codes.T.shape, dtype='float32')
  for coeff_idx in range(64):
    if training:
      apts, assignments, _, _ = uniform_quant.compute_quantization(
          raw_codes[coeff_idx, :],
          quant_multiplier*binwidths[coeff_idx],
          placement_scheme='on_zero')
      codebook.append(apts)
      quantized_codes[:, coeff_idx] = apts[assignments]
    else:
      quantized, assignments = uniform_quant.quantize(
          raw_codes[coeff_idx, :], codebook[coeff_idx],
          return_cluster_assignments=True)
      quantized_codes[:, coeff_idx] = np.copy(quantized)
    all_assignments[:, coeff_idx] = assignments

  zero_inds = cbook_inds_of_zero_pts(codebook)
  if training:
    print('Generating Huffman tables')
    huff_tab_ac, huff_tab_dc = jpeg.generate_ac_dc_huffman_tables(
        all_assignments, zero_inds)

  # compute the rate
  num_bits_count = 0
  for data_pt_idx in range(raw_codes.shape[1]):
    the_string = jpeg.generate_jpg_binary_stream(
        all_assignments[data_pt_idx], zero_inds, only_get_huffman_symbols=False,
        huffman_table_ac=huff_tab_ac, huffman_table_dc=huff_tab_dc)
    num_bits_count += len(the_string)
  total_num_code_coeffs = (raw_codes.shape[0] * raw_codes.shape[1])
  rate = num_bits_count / total_num_code_coeffs

  # now the distortion
  reconstructed_patches = np.dot(quantized_codes, dictionary.T)
  distortion = {'pSNR': compute_pSNR(target_patches, reconstructed_patches.T),
                'rMSE': compute_rMSE(target_patches, reconstructed_patches.T)}
  if fullimg_reshape_params is not None:
    full_img_gt = image_processing.assemble_image_from_patches(
        target_patches, fullimg_reshape_params['patch_dim'],
        fullimg_reshape_params['patch_positions'])
    full_img_recon = image_processing.assemble_image_from_patches(
        reconstructed_patches.T, fullimg_reshape_params['patch_dim'],
        fullimg_reshape_params['patch_positions'])
    distortion['SSIM'] = compute_ssim(full_img_gt, full_img_recon)

  if training:
    return rate, distortion, codebook, huff_tab_ac, huff_tab_dc
  else:
    return rate, distortion


def baseline_compute_RD_point(raw_codes, target_patches, dictionary,
                              quant_multiplier=None, binwidths=None,
                              precomputed_codebook=None,
                              precomputed_huff_tab1=None,
                              precomputed_huff_tab2=None,
                              fullimg_reshape_params=None):
  """
  Computes a rate-distortion pair for our Baseline quantization

  When we have yet to set the Huffman tables for the first and second halfs of
  the code, we call this without the last three parameters above. Once this
  has been computed on a training set, we can call this with those parameters
  set in order to (much more quickly) compute an RD point for some data.

  Parameters
  ----------
  raw_codes : ndarray(float) size=(s, D)
      The dataset of D samples of codes of size s
  target_patches : ndarray(float) size=(n, D)
      The D target patches for each of these codes (each has dim n)
  dictionary : ndarray(float) size=(n, s)
      The dictionary matrix that synthesizes image patches. y=Ax, and this is A
  quant_multiplier : float, optional
      This is the 'quality' parameter for jpeg. It multiplies the hifi binwidths
  binwidths : ndarray(float), optional
      These are the baseline binwidths for jpeg that we want to use for each
      coefficient of the code.
  precomputed_codebook : list(ndarray), optional
      The length s list that gives assignment points for each dimension of the
      code. The number of assignment points will be variable per dimension.
  precomputed_huff_tab1 : dictionary(str), optional
      Just a lookup table that converts symbols in the first half of the
      index message into binary string
  precomputed_huff_tab2 : dictionary(str), optional
      Just a lookup table that converts symbols in the second half of the
      index message into binary string
  fullimg_reshape_params = dictionary, optional
      If provided, then we reshape the reconstructed patches and the ground
      truth patches into a full image which we can comute the SSIM on. I
      don't believe we should be be computing the SSIM on random patches.

  Returns
  -------
  rate : float
      The number of bits per code coefficient, on average
  distortion : float
      The Peak Signal to Noise Ratio of the reconstructed patches, after
      the codes are quantized.
  codebook : list(ndarray), if precomputed_codebook is None
      The length s list that gives assignment points for each dimension of the
      code. The number of assignment points will be variable per dimension.
  huff_tab1 : dictionary(str), if precomputed_huff_tab1 is None
      Just a lookup table that converts symbols in the first half of the
      index message into binary string
  huff_tab2 : dictionary(str), if precomputed_huff_tab1 is None
      Just a lookup table that converts symbols in the second half of the
      index message into binary string
  """
  training = True if precomputed_codebook == None else False
  if training:
    codebook = []
  else:
    codebook = precomputed_codebook
    huff_tab1 = precomputed_huff_tab1
    huff_tab2 = precomputed_huff_tab2

  all_assignments = np.zeros(raw_codes.T.shape, dtype='int')
  quantized_codes = np.zeros(raw_codes.T.shape, dtype='float32')
  for coeff_idx in range(raw_codes.shape[0]):
    if training:
      apts, assignments, _, _ = uniform_quant.compute_quantization(
          raw_codes[coeff_idx, :],
          quant_multiplier*binwidths[coeff_idx],
          placement_scheme='on_zero')
      codebook.append(apts)
      quantized_codes[:, coeff_idx] = apts[assignments]
    else:
      quantized, assignments = uniform_quant.quantize(
          raw_codes[coeff_idx, :], codebook[coeff_idx],
          return_cluster_assignments=True)
      quantized_codes[:, coeff_idx] = np.copy(quantized)
    all_assignments[:, coeff_idx] = assignments

  zero_inds = cbook_inds_of_zero_pts(codebook)
  if training:
    print('Generating Huffman tables')
    huff_tab1, huff_tab2 = \
        sparse_source_coding.generate_idx_msg_huffman_tables(
            all_assignments, zero_inds)

  # compute the rate
  num_bits_count = 0
  for data_pt_idx in range(raw_codes.shape[1]):
    the_string = sparse_source_coding.generate_idx_msg_binary_stream(
        all_assignments[data_pt_idx], zero_inds, only_get_huffman_symbols=False,
        huffman_table_nz_inds=huff_tab1,
        huffman_table_nz_codeword_inds=huff_tab2)
    num_bits_count += len(the_string)
  total_num_code_coeffs = (raw_codes.shape[0] * raw_codes.shape[1])
  rate = num_bits_count / total_num_code_coeffs

  # now the distortion
  reconstructed_patches = np.dot(quantized_codes, dictionary.T)
  distortion = {'pSNR': compute_pSNR(target_patches, reconstructed_patches.T),
                'rMSE': compute_rMSE(target_patches, reconstructed_patches.T)}
  if fullimg_reshape_params is not None:
    full_img_gt = image_processing.assemble_image_from_patches(
        target_patches, fullimg_reshape_params['patch_dim'],
        fullimg_reshape_params['patch_positions'])
    full_img_recon = image_processing.assemble_image_from_patches(
        reconstructed_patches.T, fullimg_reshape_params['patch_dim'],
        fullimg_reshape_params['patch_positions'])
    distortion['SSIM'] = compute_ssim(full_img_gt, full_img_recon)

  if training:
    return rate, distortion, codebook, huff_tab1, huff_tab2
  else:
    return rate, distortion

# def Mod1_compute_RD_point(raw_codes, target_patches, dictionary,
#                           quant_multiplier=None, init_binwidths=None,
#                           precomputed_codebook=None,
#                           precomputed_codebook_lengths=None,
#                           precomputed_huff_tab1=None,
#                           precomputed_huff_tab2=None,
#                           fullimg_reshape_params=None):
#   """
#   Computes a rate-distortion pair for our Mod1 quantization
#
#   When we have yet to set the Huffman tables for the first and second halfs of
#   the code, we call this without the last three parameters above. Once this
#   has been computed on a training set, we can call this with those parameters
#   set in order to (much more quickly) compute an RD point for some data.
#
#   Parameters
#   ----------
#   raw_codes : ndarray(float) size=(s, D)
#       The dataset of D samples of codes of size s
#   target_patches : ndarray(float) size=(n, D)
#       The D target patches for each of these codes (each has dim n)
#   dictionary : ndarray(float) size=(n, s)
#       The dictionary matrix that synthesizes image patches. y=Ax, and this is A
#   quant_multiplier : float, optional
#       This is the 'quality' parameter for jpeg. It multiplies the hifi binwidths
#   init_binwidths : ndarray(float), optional
#       These are the init binwidths for Lloyd quantization
#   precomputed_codebook : list(ndarray), optional
#       The length s list that gives assignment points for each dimension of the
#       code. The number of assignment points will be variable per dimension.
#   precomputed_codebook_lengths : list(ndarray), optional
#       The length s list that gives Lloyd-computed codeword lengths for each 
#       dimension of the code. The number of assignment points will be 
#       variable per dimension.
#   precomputed_huff_tab1 : dictionary(str), optional
#       Just a lookup table that converts symbols in the first half of the
#       index message into binary string
#   precomputed_huff_tab2 : dictionary(str), optional
#       Just a lookup table that converts symbols in the second half of the
#       index message into binary string
#   fullimg_reshape_params = dictionary, optional
#       If provided, then we reshape the reconstructed patches and the ground
#       truth patches into a full image which we can comute the SSIM on. I
#       don't believe we should be be computing the SSIM on random patches.
#
#   Returns
#   -------
#   rate : float
#       The number of bits per code coefficient, on average
#   distortion : float
#       The Peak Signal to Noise Ratio of the reconstructed patches, after
#       the codes are quantized.
#   codebook : list(ndarray), if precomputed_codebook is None
#       The length s list that gives assignment points for each dimension of the
#       code. The number of assignment points will be variable per dimension.
#   huff_tab1 : dictionary(str), if precomputed_huff_tab1 is None
#       Just a lookup table that converts symbols in the first half of the
#       index message into binary string
#   huff_tab2 : dictionary(str), if precomputed_huff_tab1 is None
#       Just a lookup table that converts symbols in the second half of the
#       index message into binary string
#   """
#   training = True if precomputed_codebook == None else False
#   if training:
#     codebook = []
#     codebook_lengths = []
#   else:
#     for coeff_idx in range(raw_codes.shape[0]):
#       assert 0.0 in precomputed_codebook[coeff_idx]
#     codebook = precomputed_codebook
#     codebook_lengths = precomputed_codebook_lengths
#     huff_tab1 = precomputed_huff_tab1
#     huff_tab2 = precomputed_huff_tab2
#
#   all_assignments = np.zeros(raw_codes.T.shape, dtype='int')
#   quantized_codes = np.zeros(raw_codes.T.shape, dtype='float32')
#   for coeff_idx in range(raw_codes.shape[0]):
#     # only fit Lloyd to the nonzero values
#     nonzero_samps = np.where(raw_codes[coeff_idx] != 0.0)[0]
#     if training:
#       print('tick.')
#       inits, _, _, _ = uniform_quant.compute_quantization(
#           raw_codes[coeff_idx][nonzero_samps],
#           init_binwidths[coeff_idx],
#           placement_scheme='on_zero')
#       init_cword_len = (-1. * np.log2(1. / len(inits)) *
#                         np.ones((len(inits),)))
#       apts, assignments, _, _, cword_len = lloyd_quant.compute_quantization(
#           raw_codes[coeff_idx][nonzero_samps],
#           inits, init_cword_len, lagrange_mult=quant_multiplier, epsilon=1e-4)
#       print('tock.')
#       # This is a technical issue about having a codeword for zero
#       # in the sparse source coding module
#       closest_to_zero = np.argmin(np.abs(apts))
#       if apts[closest_to_zero] != 0.0:
#         if apts[closest_to_zero] < 0.0:
#           new_index_of_zero = closest_to_zero + 1
#         elif apts[closest_to_zero] > 0.0:
#           new_index_of_zero = closest_to_zero
#         apts = np.insert(apts, new_index_of_zero, 0.0)
#         assignments[assignments >= new_index_of_zero] = \
#           assignments[assignments >= new_index_of_zero] + 1
#         cword_len = np.insert(cword_len, new_index_of_zero, np.inf)
#         new_assignments = new_index_of_zero * np.ones(raw_codes.shape[1], dtype='int')
#         new_assignments[nonzero_samps] = assignments
#       else:
#         new_assignments = closest_to_zero * np.ones(raw_codes.shape[1], dtype='int')
#         new_assignments[nonzero_samps] = assignments
#       codebook.append(apts)
#       codebook_lengths.append(cword_len)
#       quantized_codes[:, coeff_idx] = apts[new_assignments]
#     else:
#       print('tick.')
#       zero_idx = np.where(codebook[coeff_idx] == 0.0)[0]
#       temp_cbook = np.delete(codebook[coeff_idx], zero_idx)
#       temp_cbook_lengths = np.delete(codebook_lengths[coeff_idx], zero_idx)
#       quantized, assignments = lloyd_quant.quantize(
#           raw_codes[coeff_idx][nonzero_samps], temp_cbook,
#           temp_cbook_lengths, l_weight=quant_multiplier,
#           return_cluster_assignments=True)
#       print('tock.')
#       quantized_codes[:, coeff_idx] = 0.0
#       quantized_codes[nonzero_samps, coeff_idx] = np.copy(quantized)
#       assignments[assignments >= zero_idx] = \
#         assignments[assignments >= zero_idx] + 1
#       new_assignments = zero_idx * np.ones(raw_codes.shape[1], dtype='int')
#       new_assignments[nonzero_samps] = assignments
#     all_assignments[:, coeff_idx] = new_assignments
#
#   zero_inds = cbook_inds_of_zero_pts(codebook)
#   if training:
#     print('Generating Huffman tables')
#     huff_tab1, huff_tab2 = \
#         sparse_source_coding.generate_idx_msg_huffman_tables(
#             all_assignments, zero_inds)
#
#   # compute the rate
#   num_bits_count = 0
#   for data_pt_idx in range(raw_codes.shape[1]):
#     the_string = sparse_source_coding.generate_idx_msg_binary_stream(
#         all_assignments[data_pt_idx], zero_inds, only_get_huffman_symbols=False,
#         huffman_table_nz_inds=huff_tab1,
#         huffman_table_nz_codeword_inds=huff_tab2)
#     num_bits_count += len(the_string)
#   total_num_code_coeffs = (raw_codes.shape[0] * raw_codes.shape[1])
#   rate = num_bits_count / total_num_code_coeffs
#
#   # now the distortion
#   reconstructed_patches = np.dot(quantized_codes, dictionary.T)
#   distortion = {'pSNR': compute_pSNR(target_patches, reconstructed_patches.T),
#                 'rMSE': compute_rMSE(target_patches, reconstructed_patches.T)}
#   if fullimg_reshape_params is not None:
#     full_img_gt = image_processing.assemble_image_from_patches(
#         target_patches, fullimg_reshape_params['patch_dim'],
#         fullimg_reshape_params['patch_positions'])
#     full_img_recon = image_processing.assemble_image_from_patches(
#         reconstructed_patches.T, fullimg_reshape_params['patch_dim'],
#         fullimg_reshape_params['patch_positions'])
#     distortion['SSIM'] = compute_ssim(full_img_gt, full_img_recon)
#
#   if training:
#     return rate, distortion, codebook, codebook_lengths, huff_tab1, huff_tab2
#   else:
#     return rate, distortion

# def Mod1_compute_RD_point(raw_codes, target_patches, dictionary,
#                           quant_multiplier=None, init_binwidths=None,
#                           precomputed_codebook=None,
#                           precomputed_codebook_lengths=None,
#                           precomputed_huff_tab1=None,
#                           precomputed_huff_tab2=None,
#                           fullimg_reshape_params=None):
#   """
#   Computes a rate-distortion pair for our Mod1 quantization
#
#   When we have yet to set the Huffman tables for the first and second halfs of
#   the code, we call this without the last three parameters above. Once this
#   has been computed on a training set, we can call this with those parameters
#   set in order to (much more quickly) compute an RD point for some data.
#
#   Parameters
#   ----------
#   raw_codes : ndarray(float) size=(s, D)
#       The dataset of D samples of codes of size s
#   target_patches : ndarray(float) size=(n, D)
#       The D target patches for each of these codes (each has dim n)
#   dictionary : ndarray(float) size=(n, s)
#       The dictionary matrix that synthesizes image patches. y=Ax, and this is A
#   quant_multiplier : float, optional
#       This is the 'quality' parameter for jpeg. It multiplies the hifi binwidths
#   init_binwidths : ndarray(float), optional
#       These are the init binwidths for Lloyd quantization
#   precomputed_codebook : list(ndarray), optional
#       The length s list that gives assignment points for each dimension of the
#       code. The number of assignment points will be variable per dimension.
#   precomputed_codebook_lengths : list(ndarray), optional
#       The length s list that gives Lloyd-computed codeword lengths for each 
#       dimension of the code. The number of assignment points will be 
#       variable per dimension.
#   precomputed_huff_tab1 : dictionary(str), optional
#       Just a lookup table that converts symbols in the first half of the
#       index message into binary string
#   precomputed_huff_tab2 : dictionary(str), optional
#       Just a lookup table that converts symbols in the second half of the
#       index message into binary string
#   fullimg_reshape_params = dictionary, optional
#       If provided, then we reshape the reconstructed patches and the ground
#       truth patches into a full image which we can comute the SSIM on. I
#       don't believe we should be be computing the SSIM on random patches.
#
#   Returns
#   -------
#   rate : float
#       The number of bits per code coefficient, on average
#   distortion : float
#       The Peak Signal to Noise Ratio of the reconstructed patches, after
#       the codes are quantized.
#   codebook : list(ndarray), if precomputed_codebook is None
#       The length s list that gives assignment points for each dimension of the
#       code. The number of assignment points will be variable per dimension.
#   huff_tab1 : dictionary(str), if precomputed_huff_tab1 is None
#       Just a lookup table that converts symbols in the first half of the
#       index message into binary string
#   huff_tab2 : dictionary(str), if precomputed_huff_tab1 is None
#       Just a lookup table that converts symbols in the second half of the
#       index message into binary string
#   """
#   training = True if precomputed_codebook == None else False
#   if training:
#     codebook = []
#     codebook_lengths = []
#   else:
#     codebook = precomputed_codebook
#     codebook_lengths = precomputed_codebook_lengths
#     huff_tab1 = precomputed_huff_tab1
#     huff_tab2 = precomputed_huff_tab2
#
#   all_assignments = np.zeros(raw_codes.T.shape, dtype='int')
#   quantized_codes = np.zeros(raw_codes.T.shape, dtype='float32')
#   for coeff_idx in range(raw_codes.shape[0]):
#     if training:
#       print('tick.')
#       inits, _, _, _ = uniform_quant.compute_quantization(
#           raw_codes[coeff_idx],
#           init_binwidths[coeff_idx],
#           placement_scheme='on_zero')
#       init_cword_len = (-1. * np.log2(1. / len(inits)) *
#                         np.ones((len(inits),)))
#       apts, assignments, _, _, cword_len = lloyd_quant.compute_quantization(
#           raw_codes[coeff_idx],
#           inits, init_cword_len, lagrange_mult=quant_multiplier, epsilon=1e-4)
#       print('tock.')
#       codebook.append(apts)
#       codebook_lengths.append(cword_len)
#       quantized_codes[:, coeff_idx] = apts[assignments]
#     else:
#       print('tick.')
#       quantized, assignments = lloyd_quant.quantize(
#           raw_codes[coeff_idx], codebook[coeff_idx],
#           codebook_lengths[coeff_idx], l_weight=quant_multiplier,
#           return_cluster_assignments=True)
#       print('tock.')
#       quantized_codes[:, coeff_idx] = np.copy(quantized)
#     all_assignments[:, coeff_idx] = assignments
#
#   zero_inds = cbook_inds_of_zero_pts(codebook)
#   if training:
#     print('Generating Huffman tables')
#     huff_tab1, huff_tab2 = \
#         sparse_source_coding.generate_idx_msg_huffman_tables(
#             all_assignments, zero_inds)
#
#   # compute the rate
#   num_bits_count = 0
#   for data_pt_idx in range(raw_codes.shape[1]):
#     the_string = sparse_source_coding.generate_idx_msg_binary_stream(
#         all_assignments[data_pt_idx], zero_inds, only_get_huffman_symbols=False,
#         huffman_table_nz_inds=huff_tab1,
#         huffman_table_nz_codeword_inds=huff_tab2)
#     num_bits_count += len(the_string)
#   total_num_code_coeffs = (raw_codes.shape[0] * raw_codes.shape[1])
#   rate = num_bits_count / total_num_code_coeffs
#
#   # now the distortion
#   reconstructed_patches = np.dot(quantized_codes, dictionary.T)
#   distortion = {'pSNR': compute_pSNR(target_patches, reconstructed_patches.T),
#                 'rMSE': compute_rMSE(target_patches, reconstructed_patches.T)}
#   if fullimg_reshape_params is not None:
#     full_img_gt = image_processing.assemble_image_from_patches(
#         target_patches, fullimg_reshape_params['patch_dim'],
#         fullimg_reshape_params['patch_positions'])
#     full_img_recon = image_processing.assemble_image_from_patches(
#         reconstructed_patches.T, fullimg_reshape_params['patch_dim'],
#         fullimg_reshape_params['patch_positions'])
#     distortion['SSIM'] = compute_ssim(full_img_gt, full_img_recon)
#
#   if training:
#     return rate, distortion, codebook, codebook_lengths, huff_tab1, huff_tab2
#   else:
#     return rate, distortion
#

# def Mod1_compute_RD_point(raw_codes, target_patches, dictionary,
#                           quant_multiplier=None, init_binwidths=None,
#                           precomputed_codebook=None,
#                           precomputed_codebook_lengths=None,
#                           precomputed_huff_tab1=None,
#                           fullimg_reshape_params=None):
#   """
#   Computes a rate-distortion pair for our Mod1 quantization
#
#   When we have yet to set the Huffman tables for the first and second halfs of
#   the code, we call this without the last three parameters above. Once this
#   has been computed on a training set, we can call this with those parameters
#   set in order to (much more quickly) compute an RD point for some data.
#
#   Parameters
#   ----------
#   raw_codes : ndarray(float) size=(s, D)
#       The dataset of D samples of codes of size s
#   target_patches : ndarray(float) size=(n, D)
#       The D target patches for each of these codes (each has dim n)
#   dictionary : ndarray(float) size=(n, s)
#       The dictionary matrix that synthesizes image patches. y=Ax, and this is A
#   quant_multiplier : float, optional
#       This is the 'quality' parameter for jpeg. It multiplies the hifi binwidths
#   init_binwidths : ndarray(float), optional
#       These are the init binwidths for Lloyd quantization
#   precomputed_codebook : list(ndarray), optional
#       The length s list that gives assignment points for each dimension of the
#       code. The number of assignment points will be variable per dimension.
#   precomputed_codebook_lengths : list(ndarray), optional
#       The length s list that gives Lloyd-computed codeword lengths for each 
#       dimension of the code. The number of assignment points will be 
#       variable per dimension.
#   precomputed_huff_tab1 : dictionary(str), optional
#       Just a lookup table that converts symbols in the first half of the
#       index message into binary string
#   fullimg_reshape_params = dictionary, optional
#       If provided, then we reshape the reconstructed patches and the ground
#       truth patches into a full image which we can comute the SSIM on. I
#       don't believe we should be be computing the SSIM on random patches.
#
#   Returns
#   -------
#   rate : float
#       The number of bits per code coefficient, on average
#   distortion : float
#       The Peak Signal to Noise Ratio of the reconstructed patches, after
#       the codes are quantized.
#   codebook : list(ndarray), if precomputed_codebook is None
#       The length s list that gives assignment points for each dimension of the
#       code. The number of assignment points will be variable per dimension.
#   huff_tab1 : dictionary(str), if precomputed_huff_tab1 is None
#       Just a lookup table that converts symbols in the first half of the
#       index message into binary string
#   huff_tab2 : dictionary(str), if precomputed_huff_tab1 is None
#       Just a lookup table that converts symbols in the second half of the
#       index message into binary string
#   """
#   training = True if precomputed_codebook == None else False
#   if training:
#     codebook = []
#     codebook_lengths = []
#   else:
#     codebook = precomputed_codebook
#     codebook_lengths = precomputed_codebook_lengths
#     huff_tab1 = precomputed_huff_tab1
#
#   all_assignments = np.zeros(raw_codes.T.shape, dtype='int')
#   quantized_codes = np.zeros(raw_codes.T.shape, dtype='float32')
#   for coeff_idx in range(raw_codes.shape[0]):
#     if training:
#       print('tick.')
#       inits, _, _, _ = uniform_quant.compute_quantization(
#           raw_codes[coeff_idx],
#           init_binwidths[coeff_idx],
#           placement_scheme='on_zero')
#       init_cword_len = (-1. * np.log2(1. / len(inits)) *
#                         np.ones((len(inits),)))
#       apts, assignments, _, _, cword_len = lloyd_quant.compute_quantization(
#           raw_codes[coeff_idx],
#           inits, init_cword_len, lagrange_mult=quant_multiplier, epsilon=1e-4)
#       print('tock.')
#       codebook.append(apts)
#       codebook_lengths.append(cword_len)
#       quantized_codes[:, coeff_idx] = apts[assignments]
#     else:
#       print('tick.')
#       quantized, assignments = lloyd_quant.quantize(
#           raw_codes[coeff_idx], codebook[coeff_idx],
#           codebook_lengths[coeff_idx], l_weight=quant_multiplier,
#           return_cluster_assignments=True)
#       print('tock.')
#       quantized_codes[:, coeff_idx] = np.copy(quantized)
#     all_assignments[:, coeff_idx] = assignments
#
#   zero_assign_inds = np.zeros(raw_codes.shape[0], dtype='int')
#   for coeff_idx in range(len(codebook)):
#     closest_to_zero = np.argmin(np.abs(codebook[coeff_idx]))
#     zero_assign_inds[coeff_idx] = closest_to_zero
#   rel_to_zero_assignments = all_assignments - zero_assign_inds[None, :]
#   if training:
#     print('Generating Huffman tables')
#     huff_tab1 = \
#         sparse_source_coding.generate_dense_idx_huffman_tables(
#             rel_to_zero_assignments)
#   num_bits_count = 0
#   for data_pt_idx in range(raw_codes.shape[1]):
#     binary_cw_stream = [huff_tab1[str(x)] for x in rel_to_zero_assignments[data_pt_idx]]
#     full_binary_stream = ''
#     for huff_cw in binary_cw_stream:
#       full_binary_stream += huff_cw 
#     num_bits_count += len(full_binary_stream)
#   total_num_code_coeffs = (raw_codes.shape[0] * raw_codes.shape[1])
#   rate = num_bits_count / total_num_code_coeffs
#
#   # now the distortion
#   reconstructed_patches = np.dot(quantized_codes, dictionary.T)
#   distortion = {'pSNR': compute_pSNR(target_patches, reconstructed_patches.T),
#                 'rMSE': compute_rMSE(target_patches, reconstructed_patches.T)}
#   if fullimg_reshape_params is not None:
#     full_img_gt = image_processing.assemble_image_from_patches(
#         target_patches, fullimg_reshape_params['patch_dim'],
#         fullimg_reshape_params['patch_positions'])
#     full_img_recon = image_processing.assemble_image_from_patches(
#         reconstructed_patches.T, fullimg_reshape_params['patch_dim'],
#         fullimg_reshape_params['patch_positions'])
#     distortion['SSIM'] = compute_ssim(full_img_gt, full_img_recon)
#
#   if training:
#     return rate, distortion, codebook, codebook_lengths, huff_tab1
#   else:
#     return rate, distortion

def Mod2_compute_RD_point(raw_codes, target_patches, dictionary,
                          scal_clusters, vec_cluster,
                          scal_quant_multiplier=None, scal_binwidths=None,
                          vec_quant_multiplier=None, vec_init_num_bins=None,
                          precomputed_scal_codebook=None,
                          precomputed_vec_codebook=None,
                          precomputed_vec_codebook_lengths=None,
                          precomputed_huff_tab1=None,
                          precomputed_huff_tab2=None,
                          precomputed_huff_tab3=None,
                          fullimg_reshape_params=None):
  """
  Computes a rate-distortion pair for our mod2

  When we have yet to set the Huffman tables for the first and second halfs of
  the code, we call this without the last three parameters above. Once this
  has been computed on a training set, we can call this with those parameters
  set in order to (much more quickly) compute an RD point for some data.

  Parameters
  ----------
  raw_codes : ndarray(float) size=(s, D)
      The dataset of D samples of codes of size s
  target_patches : ndarray(float) size=(n, D)
      The D target patches for each of these codes (each has dim n)
  dictionary : ndarray(float) size=(n, s)
      The dictionary matrix that synthesizes image patches. y=Ax, and this is A
  quant_multiplier : float, optional
      This is the 'quality' parameter for jpeg. It multiplies the hifi binwidths
  binwidths : ndarray(float), optional
      These are the baseline binwidths for jpeg that we want to use for each
      coefficient of the code.
  precomputed_codebook : list(ndarray), optional
      The length s list that gives assignment points for each dimension of the
      code. The number of assignment points will be variable per dimension.
  precomputed_huff_tab1 : dictionary(str), optional
      Just a lookup table that converts symbols in the first half of the
      index message into binary string
  precomputed_huff_tab2 : dictionary(str), optional
      Just a lookup table that converts symbols in the second half of the
      index message into binary string
  fullimg_reshape_params = dictionary, optional
      If provided, then we reshape the reconstructed patches and the ground
      truth patches into a full image which we can comute the SSIM on. I
      don't believe we should be be computing the SSIM on random patches.

  Returns
  -------
  rate : float
      The number of bits per code coefficient, on average
  distortion : float
      The Peak Signal to Noise Ratio of the reconstructed patches, after
      the codes are quantized.
  codebook : list(ndarray), if precomputed_codebook is None
      The length s list that gives assignment points for each dimension of the
      code. The number of assignment points will be variable per dimension.
  huff_tab1 : dictionary(str), if precomputed_huff_tab1 is None
      Just a lookup table that converts symbols in the first half of the
      index message into binary string
  huff_tab2 : dictionary(str), if precomputed_huff_tab1 is None
      Just a lookup table that converts symbols in the second half of the
      index message into binary string
  """
  training = True if precomputed_scal_codebook == None else False
  if training:
    scal_codebook = []
  else:
    scal_codebook = precomputed_scal_codebook
    vec_codebook = precomputed_vec_codebook
    vec_codebook_lengths = precomputed_vec_codebook_lengths
    huff_tab1 = precomputed_huff_tab1
    huff_tab2 = precomputed_huff_tab2
    huff_tab3 = precomputed_huff_tab3

  all_scal_assignments = np.zeros((raw_codes.shape[1], len(scal_clusters)), dtype='int')
  quantized_codes = np.zeros(raw_codes.T.shape, dtype='float32')
  for scal_coeff_idx in range(len(scal_clusters)):
    if training:
      apts, assignments, _, _ = uniform_quant.compute_quantization(
          raw_codes[scal_clusters[scal_coeff_idx], :],
          scal_quant_multiplier*scal_binwidths[scal_coeff_idx],
          placement_scheme='on_zero')
      scal_codebook.append(apts)
      quantized_codes[:, scal_clusters[scal_coeff_idx]] = apts[assignments]
    else:
      quantized, assignments = uniform_quant.quantize(
          raw_codes[scal_clusters[scal_coeff_idx], :], 
          scal_codebook[scal_coeff_idx],
          return_cluster_assignments=True)
      quantized_codes[:, scal_clusters[scal_coeff_idx]] = np.copy(quantized)
    all_scal_assignments[:, scal_coeff_idx] = assignments

  if training:
    print('Doing Lloyd vector quantization')
    print('tick.')
    init_assignments = np.random.uniform(low=-400., high=400., 
                                         size=(vec_init_num_bins, len(vec_cluster)))

    init_cword_len = (-1. * np.log2(1. / vec_init_num_bins) *
                      np.ones((vec_init_num_bins,)))
    vec_codebook, all_vec_assignments, _, _, vec_codebook_lengths = \
        lloyd_quant.compute_quantization(
          raw_codes[vec_cluster].T,
          init_assignments, init_cword_len, lagrange_mult=vec_quant_multiplier,
          epsilon=1e-4)
    print('tock.')
    quantized_codes[:, vec_cluster] = vec_codebook[all_vec_assignments]
  else:
    print('tick.')
    quantized, all_vec_assignments = lloyd_quant.quantize(
        raw_codes[vec_cluster].T,
        vec_codebook, vec_codebook_lengths, l_weight=vec_quant_multiplier,
        return_cluster_assignments=True)
    print('tock.')
    quantized_codes[:, vec_cluster] = np.copy(quantized)

  # scalar coefficients source coding
  scal_zero_inds = cbook_inds_of_zero_pts(scal_codebook)
  assert not np.any(scal_zero_inds == -1)
  if training:
    print('Generating Huffman tables for scalar coeff idx code')
    huff_tab1, huff_tab2 = \
        sparse_source_coding.generate_idx_msg_huffman_tables(
            all_scal_assignments, scal_zero_inds)

    # vector coefficients source coding
    huff_tab3 = sparse_source_coding.generate_dense_idx_huffman_tables(
        all_vec_assignments, vector_code=True)

  # compute the rate
  num_bits_count = 0
  for data_pt_idx in range(raw_codes.shape[1]):
    the_string = sparse_source_coding.generate_idx_msg_binary_stream(
        all_scal_assignments[data_pt_idx], scal_zero_inds, 
        only_get_huffman_symbols=False,
        huffman_table_nz_inds=huff_tab1,
        huffman_table_nz_codeword_inds=huff_tab2)
    the_string = the_string + huff_tab3[str(all_vec_assignments[data_pt_idx])]
    num_bits_count += len(the_string)
  total_num_code_coeffs = (raw_codes.shape[0] * raw_codes.shape[1])
  rate = num_bits_count / total_num_code_coeffs

  # now the distortion
  reconstructed_patches = np.dot(quantized_codes, dictionary.T)
  distortion = {'pSNR': compute_pSNR(target_patches, reconstructed_patches.T),
                'rMSE': compute_rMSE(target_patches, reconstructed_patches.T)}
  if fullimg_reshape_params is not None:
    full_img_gt = image_processing.assemble_image_from_patches(
        target_patches, fullimg_reshape_params['patch_dim'],
        fullimg_reshape_params['patch_positions'])
    full_img_recon = image_processing.assemble_image_from_patches(
        reconstructed_patches.T, fullimg_reshape_params['patch_dim'],
        fullimg_reshape_params['patch_positions'])
    distortion['SSIM'] = compute_ssim(full_img_gt, full_img_recon)

  if training:
    return (rate, distortion, scal_codebook, vec_codebook, vec_codebook_lengths, 
            huff_tab1, huff_tab2, huff_tab3)
  else:
    return rate, distortion

def Mod3_compute_RD_point(raw_codes, target_patches, dictionary,
                          scal_clusters, vec_cluster,
                          scal_quant_multiplier=None, scal_binwidths=None,
                          vec_quant_multiplier=None, vec_init_num_bins=None,
                          precomputed_scal_codebook=None,
                          precomputed_vec_codebook=None,
                          precomputed_vec_codebook_lengths=None,
                          precomputed_huff_tab1=None,
                          precomputed_huff_tab2=None,
                          precomputed_huff_tab3=None,
                          fullimg_reshape_params=None):
  """
  Computes a rate-distortion pair for our mod3

  When we have yet to set the Huffman tables for the first and second halfs of
  the code, we call this without the last three parameters above. Once this
  has been computed on a training set, we can call this with those parameters
  set in order to (much more quickly) compute an RD point for some data.

  Parameters
  ----------
  raw_codes : ndarray(float) size=(s, D)
      The dataset of D samples of codes of size s
  target_patches : ndarray(float) size=(n, D)
      The D target patches for each of these codes (each has dim n)
  dictionary : ndarray(float) size=(n, s)
      The dictionary matrix that synthesizes image patches. y=Ax, and this is A
  quant_multiplier : float, optional
      This is the 'quality' parameter for jpeg. It multiplies the hifi binwidths
  binwidths : ndarray(float), optional
      These are the baseline binwidths for jpeg that we want to use for each
      coefficient of the code.
  precomputed_codebook : list(ndarray), optional
      The length s list that gives assignment points for each dimension of the
      code. The number of assignment points will be variable per dimension.
  precomputed_huff_tab1 : dictionary(str), optional
      Just a lookup table that converts symbols in the first half of the
      index message into binary string
  precomputed_huff_tab2 : dictionary(str), optional
      Just a lookup table that converts symbols in the second half of the
      index message into binary string
  fullimg_reshape_params = dictionary, optional
      If provided, then we reshape the reconstructed patches and the ground
      truth patches into a full image which we can comute the SSIM on. I
      don't believe we should be be computing the SSIM on random patches.

  Returns
  -------
  rate : float
      The number of bits per code coefficient, on average
  distortion : float
      The Peak Signal to Noise Ratio of the reconstructed patches, after
      the codes are quantized.
  codebook : list(ndarray), if precomputed_codebook is None
      The length s list that gives assignment points for each dimension of the
      code. The number of assignment points will be variable per dimension.
  huff_tab1 : dictionary(str), if precomputed_huff_tab1 is None
      Just a lookup table that converts symbols in the first half of the
      index message into binary string
  huff_tab2 : dictionary(str), if precomputed_huff_tab1 is None
      Just a lookup table that converts symbols in the second half of the
      index message into binary string
  """
  training = True if precomputed_scal_codebook == None else False
  if training:
    scal_codebook = []
  else:
    scal_codebook = precomputed_scal_codebook
    vec_codebook = precomputed_vec_codebook
    vec_codebook_lengths = precomputed_vec_codebook_lengths
    huff_tab1 = precomputed_huff_tab1
    huff_tab2 = precomputed_huff_tab2
    huff_tab3 = precomputed_huff_tab3

  all_scal_assignments = np.zeros((raw_codes.shape[1], len(scal_clusters)), dtype='int')
  quantized_codes = np.zeros(raw_codes.T.shape, dtype='float32')
  for scal_coeff_idx in range(len(scal_clusters)):
    if training:
      apts, assignments, _, _ = uniform_quant.compute_quantization(
          raw_codes[scal_clusters[scal_coeff_idx], :],
          scal_quant_multiplier*scal_binwidths[scal_coeff_idx],
          placement_scheme='on_zero')
      scal_codebook.append(apts)
      quantized_codes[:, scal_clusters[scal_coeff_idx]] = apts[assignments]
    else:
      quantized, assignments = uniform_quant.quantize(
          raw_codes[scal_clusters[scal_coeff_idx], :], 
          scal_codebook[scal_coeff_idx],
          return_cluster_assignments=True)
      quantized_codes[:, scal_clusters[scal_coeff_idx]] = np.copy(quantized)
    all_scal_assignments[:, scal_coeff_idx] = assignments

  if training:
    print('Doing LSTC Lloyd vector quantization')
    print('tick.')
    init_assignments = np.random.uniform(low=-400., high=400., 
                                         size=(vec_init_num_bins, len(vec_cluster)))

    init_cword_len = (-1. * np.log2(1. / vec_init_num_bins) *
                      np.ones((vec_init_num_bins,)))
    vec_codebook, all_vec_assignments, _, _, vec_codebook_lengths = \
        lstc_lloyd_quant.compute_quantization(
          raw_codes[vec_cluster].T, dictionary[:, vec_cluster].T,
          init_assignments, init_cword_len, lagrange_mult=vec_quant_multiplier,
          epsilon=1e-4)
    print('tock.')
    quantized_codes[:, vec_cluster] = vec_codebook[all_vec_assignments]
  else:
    print('tick.')
    quantized, all_vec_assignments = lloyd_quant.quantize(
        raw_codes[vec_cluster].T, dictionary[:, vec_cluster].T,
        vec_codebook, vec_codebook_lengths, l_weight=vec_quant_multiplier,
        return_cluster_assignments=True)
    print('tock.')
    quantized_codes[:, vec_cluster] = np.copy(quantized)

  # scalar coefficients source coding
  scal_zero_inds = cbook_inds_of_zero_pts(scal_codebook)
  assert not np.any(scal_zero_inds == -1)
  if training:
    print('Generating Huffman tables for scalar coeff idx code')
    huff_tab1, huff_tab2 = \
        sparse_source_coding.generate_idx_msg_huffman_tables(
            all_scal_assignments, scal_zero_inds)

    # vector coefficients source coding
    huff_tab3 = sparse_source_coding.generate_dense_idx_huffman_tables(
        all_vec_assignments, vector_code=True)

  # compute the rate
  num_bits_count = 0
  for data_pt_idx in range(raw_codes.shape[1]):
    the_string = sparse_source_coding.generate_idx_msg_binary_stream(
        all_scal_assignments[data_pt_idx], scal_zero_inds, 
        only_get_huffman_symbols=False,
        huffman_table_nz_inds=huff_tab1,
        huffman_table_nz_codeword_inds=huff_tab2)
    the_string = the_string + huff_tab3[str(all_vec_assignments[data_pt_idx])]
    num_bits_count += len(the_string)
  total_num_code_coeffs = (raw_codes.shape[0] * raw_codes.shape[1])
  rate = num_bits_count / total_num_code_coeffs

  # now the distortion
  reconstructed_patches = np.dot(quantized_codes, dictionary.T)
  distortion = {'pSNR': compute_pSNR(target_patches, reconstructed_patches.T),
                'rMSE': compute_rMSE(target_patches, reconstructed_patches.T)}
  if fullimg_reshape_params is not None:
    full_img_gt = image_processing.assemble_image_from_patches(
        target_patches, fullimg_reshape_params['patch_dim'],
        fullimg_reshape_params['patch_positions'])
    full_img_recon = image_processing.assemble_image_from_patches(
        reconstructed_patches.T, fullimg_reshape_params['patch_dim'],
        fullimg_reshape_params['patch_positions'])
    distortion['SSIM'] = compute_ssim(full_img_gt, full_img_recon)

  if training:
    return (rate, distortion, scal_codebook, vec_codebook, vec_codebook_lengths, 
            huff_tab1, huff_tab2, huff_tab3)
  else:
    return rate, distortion
