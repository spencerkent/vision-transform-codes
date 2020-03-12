"""
Just some simple utilities for convolutional transforms
"""
import math
import torch

def get_padding_amt(image_dim, kernel_dim, dim_stride):
  leading_padding = kernel_dim - dim_stride
  trailing_padding = kernel_dim - dim_stride
  if image_dim % dim_stride != 0:
     trailing_padding += (dim_stride - (image_dim % dim_stride))
  return leading_padding, trailing_padding

def code_dim_from_padded_img_dim(padded_image_dim, kernel_dim, dim_stride):
  return 1 + int(math.ceil((padded_image_dim - kernel_dim) / dim_stride))

def create_mask(images_with_padding, padding):
  mask = torch.ones_like(images_with_padding)
  if padding is not None:
    mask[:, :, 0:padding[0][0], :] = 0.0
    mask[:, :, -padding[0][1]:, :] = 0.0
    mask[:, :, :, 0:padding[1][0]] = 0.0
    mask[:, :, :, -padding[1][1]:] = 0.0
  return mask
