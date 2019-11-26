"""
Example: Train convolutional sparse coding dictionary. These settings for 
Field natural images dset
"""

import sys
import os
examples_fullpath = os.path.dirname(os.path.abspath(__file__))
toplevel_dir_fullpath = examples_fullpath[:examples_fullpath.rfind('/')+1]
sys.path.insert(0, toplevel_dir_fullpath)

import argparse
import math
import pickle
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection as mpl_pcollection
from matplotlib.patches import Rectangle as mpl_rectangle
import torch

# from training.sparse_coding import train_dictionary as sc_train
# from utils.plotting import TrainingLivePlot
# from utils.image_processing import create_patch_training_set
from analysis_transforms import ista_convolutional
from dict_update_rules import sc_steepest_descent_convolutional
from utils import image_processing

RUN_IDENTIFIER = 'testing_convolutional_ista'

BATCH_SIZE = 1
NUM_BATCHES = 10
KERNEL_HEIGHT = 8
KERNEL_WIDTH = 8
KERNEL_STRIDE_VERT = 4
KERNEL_STRIDE_HORZ = 4
assert KERNEL_STRIDE_HORZ <= KERNEL_WIDTH
assert KERNEL_STRIDE_VERT <= KERNEL_HEIGHT
assert KERNEL_WIDTH % KERNEL_STRIDE_HORZ == 0
assert KERNEL_HEIGHT % KERNEL_STRIDE_VERT == 0

CODE_SIZE = 32

torch_device = torch.device('cuda:0')
torch.cuda.set_device(0)

# def validate_im_dim(image_dim, kernel_dim, stride_dim):
#   if kernel_dim - stride_dim != 0:
#     remainders = ((image_dim + 2 * (kernel_dim - stride_dim)) %
#                   (kernel_dim - stride_dim))
#     if np.any(remainders):
#       print('Beware, you have an image which will have some ignored pixels')
#   else:
#     # image must be evenly divided by the kernels. 
#     # TODO: deal with this more flexibly later
#     if image_dim % kernel_dim != 0:
#       raise RuntimeError('When no overlap, image must be evenly ' +
#                          'divisible into patches')
# def get_padding_amt(image_dim, kernel_dim, dim_stride):
#   leading_padding = kernel_dim - dim_stride
#   trailing_padding = kernel_dim - dim_stride - (image_dim % dim_stride)
#   return leading_padding, trailing_padding
#
# train_images = pickle.load(open('/media/expansion1/spencerkent/Datasets/Kodak/kodak_full_images_training.p', 'rb'))
# for i in range(len(train_images)):
#   assert train_images[i].ndim == 2
#   # whiten and subtract mean
#   train_images[i] = np.hstack([train_images[i], np.zeros([train_images[i].shape[0], 3])])
#   train_images[i] = train_images[i].astype('float32')
#   train_images[i] = train_images[i] / 255.
#   train_images[i] = image_processing.whiten_center_surround(train_images[i])
#   train_images[i] = train_images[i] / np.mean(train_images[i])
#   # validate_im_dim(train_images[i].shape[0], KERNEL_HEIGHT, KERNEL_STRIDE_VERT)
#   # validate_im_dim(train_images[i].shape[1], KERNEL_WIDTH, KERNEL_STRIDE_HORZ)
#
# def code_dim_from_padded_dim(padded_image_dim, kernel_dim, dim_stride):
#   return 1 + int((padded_image_dim - kernel_dim) / dim_stride)


def get_padding_amt(image_dim, kernel_dim, dim_stride):
  leading_padding = kernel_dim - dim_stride
  trailing_padding = kernel_dim - dim_stride
  if image_dim % dim_stride != 0:
     trailing_padding += (dim_stride - (image_dim % dim_stride))
  return leading_padding, trailing_padding

train_images = pickle.load(open('/media/expansion1/spencerkent/Datasets/Kodak/kodak_full_images_training.p', 'rb'))
for i in range(len(train_images)):
  assert train_images[i].ndim == 2
  # whiten and subtract mean
  train_images[i] = train_images[i].astype('float32')
  train_images[i] = train_images[i] / 255.
  train_images[i] = image_processing.whiten_center_surround(train_images[i])
  train_images[i] = train_images[i] / np.mean(train_images[i])
  # validate_im_dim(train_images[i].shape[0], KERNEL_HEIGHT, KERNEL_STRIDE_VERT)
  # validate_im_dim(train_images[i].shape[1], KERNEL_WIDTH, KERNEL_STRIDE_HORZ)


# TODO: code dim from orig dim

def get_receptive_fields(padded_image_dims, kernel_dims, kernel_strides):
  code_height = ista_convolutional.code_dim_from_padded_img_dim(
      padded_image_dims[0], kernel_dims[0], kernel_strides[0])
  code_width = ista_convolutional.code_dim_from_padded_img_dim(
      padded_image_dims[1], kernel_dims[1], kernel_strides[1])
  patch_list = []
  for y_idx in range(code_height):
    for x_idx in range(code_width):
      patch_list.append(mpl_rectangle(
        (x_idx*kernel_strides[1], y_idx*kernel_strides[0]), 
        height=kernel_dims[0], width=kernel_dims[1]))

  # rf_ypos = 0
  # while rf_ypos < padded_image_dims[0]:
  #   rf_xpos = 0
  #   while rf_xpos < padded_image_dims[1] - kernel_dims[1]:
  #     patch_list.append(mpl_rectangle((rf_xpos, rf_ypos), 
  #                                     height=kernel_dims[0],
  #                                     width=kernel_dims[1]))
  #     rf_xpos += kernel_strides[1]
  #   rf_ypos += kernel_strides[0]
  patch_collection = mpl_pcollection(patch_list, facecolor='r',
                                     edgecolor='None', alpha=0.05)
  return patch_collection

DISPLAY_WITH_RECEPTIVE_FIELDS = False
padded_train_images = []
for i in range(len(train_images)):
  pv = KERNEL_HEIGHT - KERNEL_STRIDE_VERT
  pw = KERNEL_WIDTH - KERNEL_STRIDE_HORZ
  vert_padding = get_padding_amt(train_images[i].shape[0], KERNEL_HEIGHT, 
                                 KERNEL_STRIDE_VERT)
  horz_padding = get_padding_amt(train_images[i].shape[1], KERNEL_WIDTH, 
                                 KERNEL_STRIDE_HORZ)
  padded_train_images.append(np.pad(train_images[i], 
    (vert_padding, horz_padding), mode='constant', constant_values=0.))
  if DISPLAY_WITH_RECEPTIVE_FIELDS:
    plt.subplot(121)
    plt.imshow(padded_train_images[i], cmap='Greys_r', interpolation='None')
    plt.axis('off')
    plt.gca().add_collection(get_receptive_fields(
      padded_train_images[i].shape, (KERNEL_HEIGHT, KERNEL_WIDTH), 
      (KERNEL_STRIDE_VERT, KERNEL_STRIDE_HORZ)))
    plt.subplot(122)
    remove_padding = padded_train_images[i][vert_padding[0]:-vert_padding[1],
                                            horz_padding[0]:-horz_padding[1]]
    plt.imshow(remove_padding, cmap='Greys_r', interpolation='nearest')
    plt.axis('off')
    plt.show()

# create the dictionary Tensor on the GPU
sparse_coding_dictionary = torch.randn(
    (CODE_SIZE, 3, KERNEL_HEIGHT, KERNEL_WIDTH), device=torch_device)
sparse_coding_dictionary.div_(torch.squeeze(sparse_coding_dictionary.norm(
  p=2, dim=(1, 2, 3)))[:, None, None, None])

# try to run some convSC on this image
image_on_gpu = torch.from_numpy(np.array([[padded_train_images[1],
                                           padded_train_images[2],
                                           padded_train_images[3]]])).to(torch_device)
# image_on_gpu = torch.from_numpy(padded_train_images[1][None, None, :, :]).to(torch_device)

for i in range(5):

  the_codes = ista_convolutional.run(image_on_gpu, sparse_coding_dictionary,
      kernel_stride=(KERNEL_STRIDE_VERT, KERNEL_STRIDE_HORZ),
      padding_dims=(vert_padding, horz_padding), 
      sparsity_weight=0.1, num_iters=10, nonnegative_only=True)
  input('Done with inference')

  func_val = sc_steepest_descent_convolutional.run(image_on_gpu, sparse_coding_dictionary, the_codes, (KERNEL_STRIDE_VERT, KERNEL_STRIDE_HORZ), padding_dims=(vert_padding, horz_padding), stepsize=0.001, num_iters=10, normalize_dictionary=True)
  print('Returned list: ', func_val)

  plt.figure()
  plt.title('iter ' + str(i))
  plt.plot(func_val)

plt.show()

# reconstruction = torch.nn.functional.conv_transpose2d(the_codes, 
#     sparse_coding_dictionary, stride=(KERNEL_STRIDE_VERT, KERNEL_STRIDE_HORZ))
#
# w_padding_gt = np.squeeze(image_on_gpu.cpu().numpy())[0]
# gt = w_padding_gt[vert_padding[0]:-vert_padding[1], horz_padding[0]:-horz_padding[1]]
# w_padding_recon = np.squeeze(reconstruction.cpu().numpy())[0]
# recon = w_padding_recon[vert_padding[0]:-vert_padding[1], horz_padding[0]:-horz_padding[1]]
# plt.figure()
# plt.title('Im 1')
# plt.subplot(141)
# plt.imshow(w_padding_gt, cmap='Greys_r', interpolation='None')
# plt.axis('off')
# plt.subplot(142)
# plt.axis('off')
# plt.imshow(gt, cmap='Greys_r', interpolation='None')
# plt.subplot(143)
# plt.axis('off')
# plt.imshow(w_padding_recon, cmap='Greys_r', interpolation='None')
# plt.subplot(144)
# plt.axis('off')
# plt.imshow(recon, cmap='Greys_r', interpolation='None')
#
# w_padding_gt = np.squeeze(image_on_gpu.cpu().numpy())[1]
# gt = w_padding_gt[vert_padding[0]:-vert_padding[1], horz_padding[0]:-horz_padding[1]]
# w_padding_recon = np.squeeze(reconstruction.cpu().numpy())[1]
# recon = w_padding_recon[vert_padding[0]:-vert_padding[1], horz_padding[0]:-horz_padding[1]]
# plt.figure()
# plt.title('Im 2')
# plt.subplot(141)
# plt.imshow(w_padding_gt, cmap='Greys_r', interpolation='None')
# plt.axis('off')
# plt.subplot(142)
# plt.axis('off')
# plt.imshow(gt, cmap='Greys_r', interpolation='None')
# plt.subplot(143)
# plt.axis('off')
# plt.imshow(w_padding_recon, cmap='Greys_r', interpolation='None')
# plt.subplot(144)
# plt.axis('off')
# plt.imshow(recon, cmap='Greys_r', interpolation='None')
#
# plt.figure()
# plt.title('Func val')
# plt.plot(func_val, linewidth=3)
# plt.figure()
# plt.title('l0 val')
# plt.plot(l0val, linewidth=3)
# plt.figure()
# plt.title('l1 val')
# plt.plot(l1val, linewidth=3)
# plt.figure()
# plt.title('recon val')
# plt.plot(reconval, linewidth=3)
#
# plt.show()

# starttime = time.time()
# the_codes, func_val, l0val, l1val, reconval = ista_convolutional.run(image_on_gpu, sparse_coding_dictionary,
#     kernel_stride=(KERNEL_STRIDE_VERT, KERNEL_STRIDE_HORZ),
#     padding_dims=(vert_padding, horz_padding), 
#     sparsity_weight=0.1, num_iters=100, nonnegative_only=True)
# print('Time to compute sparse code for single image: ', time.time() - starttime)
#
# reconstruction = torch.nn.functional.conv_transpose2d(the_codes, 
#     sparse_coding_dictionary, stride=(KERNEL_STRIDE_VERT, KERNEL_STRIDE_HORZ))
#
# w_padding_gt = np.squeeze(image_on_gpu.cpu().numpy())[0]
# gt = w_padding_gt[vert_padding[0]:-vert_padding[1], horz_padding[0]:-horz_padding[1]]
# w_padding_recon = np.squeeze(reconstruction.cpu().numpy())[0]
# recon = w_padding_recon[vert_padding[0]:-vert_padding[1], horz_padding[0]:-horz_padding[1]]
# plt.figure()
# plt.title('Im 1')
# plt.subplot(141)
# plt.imshow(w_padding_gt, cmap='Greys_r', interpolation='None')
# plt.axis('off')
# plt.subplot(142)
# plt.axis('off')
# plt.imshow(gt, cmap='Greys_r', interpolation='None')
# plt.subplot(143)
# plt.axis('off')
# plt.imshow(w_padding_recon, cmap='Greys_r', interpolation='None')
# plt.subplot(144)
# plt.axis('off')
# plt.imshow(recon, cmap='Greys_r', interpolation='None')
#
# w_padding_gt = np.squeeze(image_on_gpu.cpu().numpy())[1]
# gt = w_padding_gt[vert_padding[0]:-vert_padding[1], horz_padding[0]:-horz_padding[1]]
# w_padding_recon = np.squeeze(reconstruction.cpu().numpy())[1]
# recon = w_padding_recon[vert_padding[0]:-vert_padding[1], horz_padding[0]:-horz_padding[1]]
# plt.figure()
# plt.title('Im 2')
# plt.subplot(141)
# plt.imshow(w_padding_gt, cmap='Greys_r', interpolation='None')
# plt.axis('off')
# plt.subplot(142)
# plt.axis('off')
# plt.imshow(gt, cmap='Greys_r', interpolation='None')
# plt.subplot(143)
# plt.axis('off')
# plt.imshow(w_padding_recon, cmap='Greys_r', interpolation='None')
# plt.subplot(144)
# plt.axis('off')
# plt.imshow(recon, cmap='Greys_r', interpolation='None')
#
# plt.figure()
# plt.title('Func val')
# plt.plot(func_val, linewidth=3)
# plt.figure()
# plt.title('l0 val')
# plt.plot(l0val, linewidth=3)
# plt.figure()
# plt.title('l1 val')
# plt.plot(l1val, linewidth=3)
# plt.figure()
# plt.title('recon val')
# plt.plot(reconval, linewidth=3)
#
# plt.show()
#
