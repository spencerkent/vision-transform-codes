"""
Utilities for creating image datasets to train a vision transform code
"""
import pickle
import numpy as np
import scipy.io
import h5py

from utils import image_processing as ip_util

def create_patch_training_set(order_of_preproc_ops, patch_dimensions,
    batch_size, num_batches, edge_buffer, dataset, datasetparams,
    flatten_patches=True):
  """
  Creates a dataset of image patches. Can be 'flattened' or not.

  Flattened images are used to train the 'fully-connected transform codes while
  unflattened patches (which have a height, width, and channel dimension) are
  used to train convolutional transform codes.

  Parameters
  ----------
  order_of_preproc_ops : list(str)
      Specifies the preprocessing operations to perform on the data. Currently
      available operations are
      ----
      {'whiten_center_surround', 'whiten_ZCA', 'patch', 'pad',
       'center_each_component', 'center_each_patch',
       'normalize_component_variance', 'divide_by_constant',
       'shift_by_constant', 'local_contrast_nomalization'}.
      ----
      Example: (we want to perform Bruno's whitening in the fourier domain and
                also have the components of each patch have zero mean)
          ['whiten_center_surround', 'patch', 'center_each_component']
  patch_dimensions : tuple(int, int)
      The size in pixels of each patch
  batch_size : int
      The number of patches in a batch
  num_batches : int
      The total number of batches to assemble
  edge_buffer : int
      The buffer from the edge of the image from which we will not include any
      patches.
  dataset : str
      The name of the dataset to grab patches from. Currently one of
      {'Field_NW', 'vanHateren', 'Kodak_BW'}.
  datasetparams : dictionary
      A dictionary of parameters that may be specific to the dataset. Currently
      just specifies the filepath of the data file and which images to
      exclude from the training set.
      'filepath' : str
      'exclude' : list(int)
      'div_constant': float (only checked if divide_by_constant preproc op)
      'shift_constant': float (only checked if shift_by_constant preproc op)
      'padding': tuple(tuple) (only checked in pad preproc op)
  flatten_patches : bool
      Flatten each patch into a vector, using the default behavior of Numpy's
      .reshape() method. Used for training 'fully-connected' transform codes.
      If False, we leave patches with height, width, and channel dimensions.
      Default True.

  Returns
  -------
  return_dict : dictionary
    'batched_patches' : ndarray(float32, size=(k, n, b) OR (k, b, pc, ph, pw))
        The patches training set where pc=number of color channels in a patch,
        ph=patch_dimensions[0], pw=patch_dimensions[1], k=num_batches,
        b=batch_size, and n=pc*ph*pw. These are patches ready to be sent to
        the gpu and consumed in PyTorch
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
    'ZCA_parameters : dictionary, optional
        See whiten_ZCA() above.
  """
  assert 'patch' in order_of_preproc_ops
  # our convention is that the first axis indexes images
  if dataset == 'Field_NW':
    unprocessed_images = scipy.io.loadmat(
      datasetparams['filepath'])['IMAGESr'].astype('float32')
    temp = np.transpose(unprocessed_images, (2, 0, 1))
    unprocessed_images = [temp[x][:, :, None] for x in range(temp.shape[0])]
  elif dataset == 'vanHateren':
    # this dataset is MUCH larger so it will take some time to load into memory
    with h5py.File(datasetparams['filepath']) as file_handle:
      temp = np.array(file_handle['van_hateren_good'], dtype='float32')
    unprocessed_images = [temp[x][:, :, None] for x in range(temp.shape[0])]
    #^ maximum pixel value is 1.0, but minimum value is NOT 0.0, more like 0.4
  elif dataset == 'Kodak_BW':
    unprocessed_images = pickle.load(open(datasetparams['filepath'], 'rb'))
    # this is a list of images, so that each image can be EITHER
    # 752x496 or 496x752, whichever makes it upright
    unprocessed_images = [x.astype('float32')[:, :, None]
                          for x in unprocessed_images]
    #^ these should be float32 for further processing...
  elif dataset == 'Kodak':
    raise NotImplementedError('This is next')
  else:
    raise KeyError('Unrecognized dataset ' + dataset)
  eligible_image_inds = np.array([x for x in range(len(unprocessed_images))
                                  if x not in datasetparams['exclude']])

  p_imgs = [unprocessed_images[x] for x in eligible_image_inds]
  num_color_channels = p_imgs[0].shape[2]
  already_patched_flag = False
  for preproc_op in order_of_preproc_ops:

    if preproc_op == 'patch':
      max_vert_pos = []
      min_vert_pos = []
      max_horz_pos = []
      min_horz_pos = []
      for img_idx in range(len(p_imgs)):
        max_vert_pos.append(
            p_imgs[img_idx].shape[0] - patch_dimensions[0] - edge_buffer)
        min_vert_pos.append(edge_buffer)
        max_horz_pos.append(
            p_imgs[img_idx].shape[1] - patch_dimensions[1] - edge_buffer)
        min_horz_pos.append(edge_buffer)
      num_imgs = len(p_imgs)

      all_patches = np.zeros(
        [num_batches*batch_size, patch_dimensions[0], patch_dimensions[1],
         num_color_channels], dtype='float32')

      p_idx = 0
      for batch_idx in range(num_batches):
        for _ in range(batch_size):
          img_idx = np.random.randint(low=0, high=num_imgs)
          vert_pos = np.random.randint(low=min_vert_pos[img_idx],
                                       high=max_vert_pos[img_idx])
          horz_pos = np.random.randint(low=min_horz_pos[img_idx],
                                       high=max_horz_pos[img_idx])
          all_patches[p_idx] = p_imgs[img_idx][
            vert_pos:vert_pos+patch_dimensions[0],
            horz_pos:horz_pos+patch_dimensions[1]]
          p_idx += 1
        if batch_idx % 1000 == 0 and batch_idx != 0:
          print('Finished creating', batch_idx, 'batches')
      print('Done.')
      already_patched_flag = True

    # TODO: change im proc to avoid having to reshape these each time
    elif preproc_op == 'whiten_center_surround':
      if already_patched_flag:
        raise KeyError('We typically preform this type of whitening before ' +
                       'patching the images')
      for img_idx in range(len(p_imgs)):
        p_imgs[img_idx] = ip_util.whiten_center_surround(p_imgs[img_idx])

    elif preproc_op == 'whiten_ZCA':
      if not already_patched_flag:
        raise KeyError('You ought to patch image before trying to compute a ' +
                       'ZCA whitening transform')
      temp_flat, computed_ZCA_params = ip_util.whiten_ZCA(
          np.reshape(all_patches, (all_patches.shape[0], -1)).T)
      all_patches = np.reshape(temp_flat.T, all_patches.shape)

    elif preproc_op == 'local_contrast_normalization':
      if already_patched_flag:
        raise KeyError('We typically preform this before ' +
                       'patching the images')
      for img_idx in range(len(p_imgs)):
        p_imgs[img_idx] = ip_util.local_contrast_nomalization(
            p_imgs[img_idx], kernel_size=datasetparams['lcn_kernel_sz'])

    elif preproc_op == 'divide_by_constant':
      if already_patched_flag:
        all_patches = all_patches / datasetparams['div_constant']
      else:
        # likely use case is to do this before whitening and patching
        for img_idx in range(len(p_imgs)):
          p_imgs[img_idx] = p_imgs[img_idx] / datasetparams['div_constant']

    elif preproc_op == 'shift_by_constant':
      if already_patched_flag:
        all_patches = all_patches + datasetparams['shift_constant']
      else:
        for img_idx in range(len(p_imgs)):
          p_imgs[img_idx] = p_imgs[img_idx] + datasetparams['shift_constant']

    elif preproc_op == 'center_each_component':
      if not already_patched_flag:
        raise KeyError('You ought to patch the data before trying to ' +
                       'center each component')
      temp_flat, orig_means = ip_util.center_each_component(
          np.reshape(all_patches, (all_patches.shape[0], -1)).T)
      all_patches = np.reshape(temp_flat.T, all_patches.shape)

    elif preproc_op == 'normalize_component_variance':
      if not already_patched_flag:
        raise KeyError('You ought to patch the data before normalizing it')
      temp_flat, orig_variances = ip_util.normalize_component_variance(
          np.reshape(all_patches, (all_patches.shape[0], -1)).T)
      all_patches = np.reshape(temp_flat.T, all_patches.shape)

    elif preproc_op == 'center_each_patch':
      if not already_patched_flag:
        raise KeyError('You ought to patch the data before trying to ' +
                       'center each patch')
      temp_flat, _ = ip_util.center_each_sample(
          np.reshape(all_patches, (all_patches.shape[0], -1)).T)
      all_patches = np.reshape(temp_flat.T, all_patches.shape)

    elif preproc_op == 'pad':
      if not already_patched_flag:
        raise KeyError('You ought to patch the data first. Padding is added '+
                       'to the patches')
      if flatten_patches:
        raise KeyError('Flattend patches shouldnt require padding')
      all_patches = np.pad(all_patches, 
          ((0, 0),) + datasetparams['padding'] + ((0, 0),), mode='constant',
          constant_values=0.)
    else:
      raise KeyError('Unrecognized preprocessing op ' + preproc_op)

  # now we finally chunk this up into batches and return
  if flatten_patches:
    return_dict = {'batched_patches': np.moveaxis(
      all_patches.reshape((num_batches, batch_size, -1)), 1, 2)}
  else:
    # torch uses a color-channel-first convention so we reshape to reflect this
    return_dict = {'batched_patches': np.moveaxis(
      all_patches.reshape((num_batches, batch_size) + all_patches.shape[1:]),
      4, 2)}

  if 'center_each_component' in order_of_preproc_ops:
    return_dict['original_component_means'] = orig_means
  if 'normalize_component_variance' in order_of_preproc_ops:
    return_dict['original_component_variances'] = orig_variances
  if 'whiten_ZCA' in order_of_preproc_ops:
    return_dict['ZCA_parameters'] = computed_ZCA_params

  return return_dict
