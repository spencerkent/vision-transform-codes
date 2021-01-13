"""
Generate some test datasets, will be used by other tests"
"""
import _set_the_path

import pickle

from utils.dataset_generation import create_patch_training_set
from utils.convolutions import get_padding_amt
from utils import defaults

# Field data, vanilla whitened center-surround, 16x16
trn_val_dsets = {
    'training': create_patch_training_set(
      num_samples=10000,
      patch_dimensions=(16, 16), edge_buffer=5,
      dataset='Field_NW', order_of_preproc_ops=[
        'standardize_data_range', 'whiten_center_surround', 'patch']),
    'validation': create_patch_training_set(
      num_samples=5000,
      patch_dimensions=(16, 16), edge_buffer=5,
      dataset='Field_NW', order_of_preproc_ops=[
        'standardize_data_range', 'whiten_center_surround', 'patch'])}
pickle.dump(trn_val_dsets, open(defaults.dataset_directory /
  'vtc_testing/field_white_16x16.p', 'wb'))

# ZCA-whitened 8x12
trn_val_dsets = {
    'training': create_patch_training_set(
      num_samples=10000,
      patch_dimensions=(8, 12), edge_buffer=3,
      dataset='Kodak_BW', order_of_preproc_ops=[
        'standardize_data_range', 'patch', 'whiten_ZCA']),
    'validation': create_patch_training_set(
      num_samples=5000,
      patch_dimensions=(8, 12), edge_buffer=3,
      dataset='Kodak_BW', order_of_preproc_ops=[
        'standardize_data_range', 'patch', 'whiten_ZCA'])}
pickle.dump(trn_val_dsets, open(defaults.dataset_directory /
  'vtc_testing/ZCA_8x12.p', 'wb'))

# Field data, vanilla whitened center-surround, big patches, (for conv)
# 256x256 patches, padding designed for 16x16 kernels with stride 8
vert_padding = get_padding_amt(256, 16, 8)
horz_padding = get_padding_amt(256, 16, 8)
trn_val_dsets = {
    'training': create_patch_training_set(
      num_samples=20,
      patch_dimensions=(256, 256), edge_buffer=5,
      dataset='Field_NW', order_of_preproc_ops=[
        'standardize_data_range', 'whiten_center_surround', 'patch', 'pad'],
      extra_params={'padding': (vert_padding, horz_padding),
                    'flatten_patches': False}),
    'validation': create_patch_training_set(
      num_samples=10,
      patch_dimensions=(256, 256), edge_buffer=5,
      dataset='Field_NW', order_of_preproc_ops=[
        'standardize_data_range', 'whiten_center_surround', 'patch', 'pad'],
      extra_params={'padding': (vert_padding, horz_padding),
                    'flatten_patches': False})}

pickle.dump(trn_val_dsets, open(defaults.dataset_directory /
  'vtc_testing/field_white_256x256_w_padding.p', 'wb'))

# # LCN with padding 32x32
# # ** slow, applies LCN to each of the full images regardless of DSET size.
# # ** Commented out for sped
# trn_val_dsets = {
#     'training': create_patch_training_set(
#       num_samples=10,
#       patch_dimensions=(156, 156), edge_buffer=20,
#       dataset='Field_NW', order_of_preproc_ops=[
#         'standardize_data_range', 'local_contrast_normalization',
#         'patch', 'pad'],
#       extra_params={'padding': ((5, 5), (5, 5)), 'lcn_filter_sigma' : 6,
#                     'flatten_patches': False}),
#     'validation': create_patch_training_set(
#       num_samples=50,
#       patch_dimensions=(156, 156), edge_buffer=20,
#       dataset='Field_NW', order_of_preproc_ops=[
#         'standardize_data_range', 'local_contrast_normalization',
#         'patch', 'pad'],
#       extra_params={'padding': ((5, 5), (5, 5)), 'lcn_filter_sigma' : 6,
#                     'flatten_patches': False})}
# pickle.dump(trn_val_dsets, open(defaults.dataset_directory /
#   'vtc_testing/LCN_pad_32x32.p', 'wb'))
