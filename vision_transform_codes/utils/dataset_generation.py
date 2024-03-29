"""
Utilities for creating image datasets to train a vision transform code
"""
import pickle
import numpy as np
import scipy.io
import h5py
import torch

from utils import image_processing as ip_util
from utils import defaults

class OneOutputDset(torch.utils.data.Dataset):
  """Just like torch.utils.data.TensorDataset, but doesn't return a tuple"""
  def __init__(self, single_tensor):
    self.tensor = single_tensor
  def __getitem__(self, index):
    return self.tensor[index]
  def __len__(self):
    return self.tensor.size(0)

def create_patch_training_set(num_samples, patch_dimensions, edge_buffer,
                              dataset, order_of_preproc_ops, extra_params={}):
  """
  Creates a dataset of image patches.

  Patches can be 'flattened' or not. Flattened patches are used to train
  'fully-connected' transform codes, while unflattened patches (which have a
  height, width, and channel dimension) are used to train convolutional
  transform codes.

  Parameters
  ----------
  num_samples : int
      The total number of samples to draw. Batching is handled separately from
      this function
  patch_dimensions : tuple(int, int)
      The size in pixels of each patch
  edge_buffer : int
      The buffer from the edge of the image from which we will not include any
      patches.
  dataset : str
      The name of the dataset to grab patches from. Currently one of
      {'Field_NW', 'vanHateren', 'Kodak_BW'}.
  order_of_preproc_ops : list(str)
      Specifies the preprocessing operations to perform on the data. Currently
      available operations are
      ----
      {'standardize_data_range', 'whiten_center_surround', 'whiten_ZCA',
       'patch', 'pad', 'center_each_component', 'center_each_patch',
       'normalize_component_variance', 'local_contrast_normalization',
       'local_luminance_subtraction'}.
  extra_params : dictionary, optional
      A dictionary of extra parameters, usually specific to the dataset or the
      preprocessing operations. Can, for instance, specify a filepath for the
      raw images (to be used instead of the default), certain images to
      exclude, or some padding to add to the patches.
        'filepath' : str, optional
        'exclude' : list(int), optional
        'padding': tuple(tuple), optional (only checked in pad preproc op)
        'lcn_filter_sigma' : tuple(int, int), optional
        'flatten_patches' : bool, optional
          Flatten each patch into a vector, using the default behavior of
          NumPy's .reshape() method. Used for training 'fully-connected'
          transform codes. If False, we leave patches with height, width, and
          channel dimensions. Default True.

  Returns
  -------
  return_dict : dictionary
    'patches' : ndarray(float32, size=(d, n) OR (d, pc, ph, pw))
        The patches training set where pc=number of color channels in a patch,
        ph=patch_dimensions[0], pw=patch_dimensions[1], d=num_patches and
        n=pc*ph*pw. These are patches ready to be sent to the gpu and
        consumed in PyTorch directly, or wrapped in a Dataset and Dataloader
        to facilitate batching, shuffling, etc.
    --- OPTIONAL RETURNS ---
    'original_component_means' :
      ndarray(float32, size=(n,) OR (pc, ph, pw)), optional
        If 'center_each_component' was requested as a preprocessing step we
        return the original component means so that future data can be
        processed using this same exact centering
    'original_component_variances' :
      ndarray(float32, size=(n,) OR (pc, ph, pw)), optional
        If 'normalize_component_variance' was requested as a preprocessing step
        we return the original variances so that future data can be processed
        using this same exact normalization
    'original_data_range' : tuple(float, float), optional
        #TODO add docstring
    'ZCA_parameters : dictionary, optional
        See whiten_ZCA() above.
  """
  assert 'patch' in order_of_preproc_ops
  if 'pad' in order_of_preproc_ops:
    assert 'padding' in extra_params
  if 'local_contrast_normalization' in order_of_preproc_ops:
    assert 'lcn_filter_sigma' in extra_params
  if 'local_luminance_subtraction' in order_of_preproc_ops:
    assert 'lls_filter_sigma' in extra_params
  if 'standardize_data_range' in order_of_preproc_ops:
    idx_sdr = [x for x in range(len(order_of_preproc_ops))
               if order_of_preproc_ops[x] == 'standardize_data_range']
    assert len(idx_sdr) == 1 and idx_sdr[0] == 0
  if 'filepath' in extra_params:
    filepath = extra_params['filepath']
  else:
    filepath = defaults.raw_data_filepaths[dataset]
  if 'flatten_patches' in extra_params:
    flatten_patches = extra_params['flatten_patches']
  else:
    flatten_patches = True  # default behavior is to flatten
  if 'whiten_center_surround' in order_of_preproc_ops:
    if 'whitening_cutoff_low' in extra_params:
      wcl = extra_params['whitening_cutoff_low']
    else:
      wcl = 1e-3
    if 'whitening_cutoff_high' in extra_params:
      wch = extra_params['whitening_cutoff_high']
    else:
      wch = 0.9


  # pre_patch_imgs is the main datastructure. List so that (pre-patch) images
  # can have different dimensions. Always casted to float32.
  if dataset == 'Field_NW':
    unprocessed_images = scipy.io.loadmat(
        filepath)['IMAGESr'].astype('float32')
    # ^have kind of a weird range, they're in between ~-3.19 and ~6.4, but
    #  not every image maxes out the range. Unclear what Bruno did to produce
    #  these images...
    temp = np.transpose(unprocessed_images, (2, 0, 1))
    pre_patch_imgs = [temp[x][:, :, None] for x in range(temp.shape[0])]
  elif dataset == 'vanHateren':
    # this dataset is MUCH larger so it will take some time to load into memory
    print('Im just trying to load data...')
    with h5py.File(filepath) as file_handle:
      temp = np.array(file_handle['van_hateren_good'], dtype='float32')
    # ^maximum pixel value is 1.0, but minimum value is NOT 0.0, more like 0.4
    #  some notes on this dataset here: http://bethgelab.org/datasets/vanhateren/
    #  but note that these were curated and further processed by Dylan Paiton.
    pre_patch_imgs = [temp[x][:, :, None] for x in range(temp.shape[0])]
  elif dataset == 'Kodak_BW':
    unprocessed_images = pickle.load(open(filepath, 'rb'))
    # ^These are [0, 255] uint8's, thankfully. Weirdly though, each image maxes
    #  out this range so it's as if they've all been renormalized. This is one
    #  way that this dataset is not 'natural'.
    pre_patch_imgs = [x.astype('float32')[:, :, None]
                      for x in unprocessed_images]
  elif dataset == 'Kodak':
    raise NotImplementedError('This is next')
  else:
    raise KeyError('Unrecognized dataset ' + dataset)

  if 'exclude' in extra_params:
    pre_patch_imgs = [pre_patch_imgs[x] for x in range(len(pre_patch_imgs))
                      if x not in extra_params['exclude']]

  if 'local_contrast_normalization' in order_of_preproc_ops:
    image_local_contrasts = [np.zeros(pre_patch_imgs[x].shape, dtype='float32')
                             for x in range(len(pre_patch_imgs))]
  if 'local_luminance_subtraction' in order_of_preproc_ops:
    image_local_luminances = [np.zeros(pre_patch_imgs[x].shape, dtype='float32')
                              for x in range(len(pre_patch_imgs))]

  num_color_channels = pre_patch_imgs[0].shape[2]
  already_patched_flag = False
  for preproc_op in order_of_preproc_ops:

    if preproc_op == 'standardize_data_range':
      # what we mean by standardize is simply to take the maximum value across
      # the entire dataset and make this 1.0, and to take the minimum value
      # and make it zero. This will leave individual images with the same
      # relative luminances and constrasts, but just put the data in a
      # standard range.
      min_val = np.min([np.min(pre_patch_imgs[x])
                        for x in range(len(pre_patch_imgs))])
      max_val = np.max([np.max(pre_patch_imgs[x])
                        for x in range(len(pre_patch_imgs))])
      assert max_val > min_val
      for img_idx in range(len(pre_patch_imgs)):
        pre_patch_imgs[img_idx] = ((pre_patch_imgs[img_idx] - min_val) /
                                   (max_val - min_val))

    elif preproc_op == 'patch':
      max_vert_pos = []
      min_vert_pos = []
      max_horz_pos = []
      min_horz_pos = []
      for img_idx in range(len(pre_patch_imgs)):
        max_vert_pos.append(pre_patch_imgs[img_idx].shape[0] -
            patch_dimensions[0] - edge_buffer)
        min_vert_pos.append(edge_buffer)
        max_horz_pos.append(pre_patch_imgs[img_idx].shape[1] -
            patch_dimensions[1] - edge_buffer)
        min_horz_pos.append(edge_buffer)
      num_imgs = len(pre_patch_imgs)

      all_patches = np.zeros(
        [num_samples, patch_dimensions[0], patch_dimensions[1],
         num_color_channels], dtype='float32')
      if 'local_contrast_normalization' in order_of_preproc_ops:
        all_patches_contrast = np.zeros(
          [num_samples, patch_dimensions[0], patch_dimensions[1],
           num_color_channels], dtype='float32')
      if 'local_luminance_subtraction' in order_of_preproc_ops:
        all_patches_luminance = np.zeros(
          [num_samples, patch_dimensions[0], patch_dimensions[1],
           num_color_channels], dtype='float32')

      for p_idx in range(num_samples):
        img_idx = np.random.randint(low=0, high=num_imgs)
        vert_pos = np.random.randint(low=min_vert_pos[img_idx],
                                     high=max_vert_pos[img_idx])
        horz_pos = np.random.randint(low=min_horz_pos[img_idx],
                                     high=max_horz_pos[img_idx])
        all_patches[p_idx] = pre_patch_imgs[img_idx][
          vert_pos:vert_pos+patch_dimensions[0],
          horz_pos:horz_pos+patch_dimensions[1]]
        if 'local_contrast_normalization' in order_of_preproc_ops:
          all_patches_contrast[p_idx] = image_local_contrasts[img_idx][
            vert_pos:vert_pos+patch_dimensions[0],
            horz_pos:horz_pos+patch_dimensions[1]]
        if 'local_luminance_subtraction' in order_of_preproc_ops:
          all_patches_luminance[p_idx] = image_local_luminances[img_idx][
            vert_pos:vert_pos+patch_dimensions[0],
            horz_pos:horz_pos+patch_dimensions[1]]
        if p_idx % 100000 == 0 and p_idx != 0:
          print('Finished creating', p_idx, 'samples')
      already_patched_flag = True

    elif preproc_op == 'whiten_center_surround':
      if already_patched_flag:
        raise KeyError('We typically preform this type of whitening before ' +
                       'patching the images')
      for img_idx in range(len(pre_patch_imgs)):
        pre_patch_imgs[img_idx] = ip_util.whiten_center_surround(
            pre_patch_imgs[img_idx], cutoffs={'low': wcl, 'high': wch},
            norm_and_threshold=False)

    elif preproc_op == 'whiten_ZCA':
      if not already_patched_flag:
        raise KeyError('You ought to patch image before trying to compute a ' +
                       'ZCA whitening transform')
      temp_flat, computed_ZCA_params = ip_util.whiten_ZCA(
          np.reshape(all_patches, (all_patches.shape[0], -1)))
      all_patches = np.reshape(temp_flat, all_patches.shape)

    elif preproc_op == 'local_contrast_normalization':
      if already_patched_flag:
        raise KeyError('We typically preform this before ' +
                       'patching the images')
      for img_idx in range(len(pre_patch_imgs)):
        pre_patch_imgs[img_idx], image_local_contrasts[img_idx] = \
            ip_util.local_contrast_normalization(
            pre_patch_imgs[img_idx],
            filter_sigma=extra_params['lcn_filter_sigma'],
            return_normalizer=True)

    elif preproc_op == 'local_luminance_subtraction':
      if already_patched_flag:
        raise KeyError('We typically preform this before ' +
                       'patching the images')
      for img_idx in range(len(pre_patch_imgs)):
        pre_patch_imgs[img_idx], image_local_luminances[img_idx] = \
            ip_util.local_luminance_subtraction(
            pre_patch_imgs[img_idx],
            filter_sigma=extra_params['lls_filter_sigma'],
            return_subtractor=True)

    elif preproc_op == 'center_each_component':
      if not already_patched_flag:
        raise KeyError('You ought to patch the data before trying to ' +
                       'center each component')
      temp_flat, orig_means = ip_util.center_each_component(
          np.reshape(all_patches, (all_patches.shape[0], -1)))
      all_patches = np.reshape(temp_flat, all_patches.shape)

    elif preproc_op == 'normalize_component_variance':
      if not already_patched_flag:
        raise KeyError('You ought to patch the data before normalizing it')
      temp_flat, orig_variances = ip_util.normalize_component_variance(
          np.reshape(all_patches, (all_patches.shape[0], -1)))
      all_patches = np.reshape(temp_flat, all_patches.shape)

    elif preproc_op == 'center_each_patch':
      if not already_patched_flag:
        raise KeyError('You ought to patch the data before trying to ' +
                       'center each patch')
      temp_flat, _ = ip_util.center_each_sample(
          np.reshape(all_patches, (all_patches.shape[0], -1)))
      all_patches = np.reshape(temp_flat, all_patches.shape)

    elif preproc_op == 'pad':
      if not already_patched_flag:
        raise KeyError('You ought to patch the data first. Padding is added '+
                       'to the patches')
      if flatten_patches:
        raise KeyError('Flattened patches shouldnt require padding')
      all_patches = np.pad(all_patches,
          ((0, 0),) + extra_params['padding'] + ((0, 0),), mode='constant',
          constant_values=0.)
      if 'local_contrast_normalization' in order_of_preproc_ops:
        all_patches_contrast = np.pad(all_patches_contrast,
            ((0, 0),) + extra_params['padding'] + ((0, 0),), mode='constant',
            constant_values=0.)
      if 'local_luminance_subtraction' in order_of_preproc_ops:
        all_patches_luminance = np.pad(all_patches_luminance,
            ((0, 0),) + extra_params['padding'] + ((0, 0),), mode='constant',
            constant_values=0.)
    else:
      raise KeyError('Unrecognized preprocessing op ' + preproc_op)

  if flatten_patches:
    return_dict = {'patches': all_patches.reshape((num_samples, -1))}
    if 'local_contrast_normalization' in order_of_preproc_ops:
      return_dict['local_contrasts'] = all_patches_contrast.reshape(
          (num_samples, -1))
    if 'local_luminance_subtraction' in order_of_preproc_ops:
      return_dict['local_luminances'] = all_patches_luminance.reshape(
          (num_samples, -1))
  else:
    # torch uses a color-channel-first convention so we reshape to reflect this
    return_dict = {'patches': np.moveaxis(all_patches, 3, 1)}
    if 'local_contrast_normalization' in order_of_preproc_ops:
      return_dict['local_contrasts'] = np.moveaxis(all_patches_contrast, 3, 1)
    if 'local_luminance_subtraction' in order_of_preproc_ops:
      return_dict['local_luminances'] = np.moveaxis(
          all_patches_luminance, 3, 1)

  if 'center_each_component' in order_of_preproc_ops:
    return_dict['original_component_means'] = orig_means
  if 'normalize_component_variance' in order_of_preproc_ops:
    return_dict['original_component_variances'] = orig_variances
  if 'whiten_ZCA' in order_of_preproc_ops:
    return_dict['ZCA_parameters'] = computed_ZCA_params

  return return_dict
