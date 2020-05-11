"""
Some utilities for wrangling image data.

These may be relevant for implementing sparse coding or other transform coding
strategies on images. Just leaving a few relevant references:

.. [1] Olshausen, B. A., & Field, D. J. (1997). Sparse coding with an
       overcomplete basis set: A strategy employed by V1?. Vision research,
       37(23), 3311-3325.
"""
import numpy as np
from scipy.signal import convolve2d
from matplotlib import pyplot as plt


def filter_sd(image, filter_spatial):
  """
  Filters an image using a filter specified in the {s}patial {d}omain

  Parameters
  ----------
  image : ndarray(float32 or uint8, size=(h, w, c))
      The image to be filtered. The filter is applied to each color
      channel independently.
  filter_spatial : ndarray(float32 or uint8, size=(fh, fw))
      The filter has fh samples in the vertical dimension and fw samples in
      the horizontal dimension

  Returns
  -------
  filtered_image : ndarray(float32, size=(h, w, c))
  """
  assert image.dtype in ['float32', 'uint8']
  filtered_image = np.zeros(image.shape, dtype='float32')
  for color_channel in range(image.shape[2]):
    filtered_image[:, :, color_channel] = convolve2d(
        image[:, :, color_channel], filter_spatial, 'same', boundary='symm')
    #^ using the 'symmetric' boundary condition seems to reduce artifacts at
    #  the image boundary. Worth looking into more closely.
  return filtered_image


def filter_fd(image, filter_DFT):
  """
  Filters an image using a filter specified in the {f}requency {d}omain

  This may optionally pad the image so as to match the number of samples in the
  filter DFT. We should make sure this is greater than or equal to the size of
  the image.

  Parameters
  ----------
  image : ndarray(float32 or uint8, size=(h, w, c))
      The image to be filtered. The filter is applied to each color
      channel independently.
  filter_DFT : ndarray(complex128, size=(fh, fw))
      The filter has fh samples in the vertical dimension and fw samples in
      the horizontal dimension

  Returns
  -------
  filtered_image : ndarray(float32, size=(h, w, c))
  """
  assert image.dtype in ['float32', 'uint8']
  assert filter_DFT.shape[0] >= image.shape[0], "don't undersample DFT"
  assert filter_DFT.shape[1] >= image.shape[1], "don't undersample DFT"
  filtered_image = np.zeros(image.shape, dtype='float32')
  for color_channel in range(image.shape[2]):
    filtered_image[:, :, color_channel] = np.real(np.fft.ifft2(
      filter_DFT * np.fft.fft2(image[:, :, color_channel], filter_DFT.shape),
      filter_DFT.shape)).astype('float32')[0:image.shape[0], 0:image.shape[1]]
  return filtered_image


def get_low_pass_filter(DFT_num_samples, filter_parameters,
                        norm_and_threshold=True):
  """
  Returns the DFT of a lowpass filter that can be applied to an image

  Parameters
  ----------
  DFT_num_samples : (int, int)
      The number of samples on the DFT vertical ax and the DFT horizontal ax
  filter_parameters : dictionary
      Parameters of the filter. These may vary depending on the filter shape.
      Smoother filters reduce ringing artifacts but can throw away a lot of
      information in the middle frequencies. For now we just have 'exponential'
      but may add other filter shapes in the future. Each shape has some
      specific parameters.
      'shape' : str
        One of {'exponential'}
      ** if 'shape' == 'exponential', then also:
      'cutoff' : float \in [0, 1]
          A fraction of the 2d nyquist frequency at which to set the cutoff.
      'order' : float \in [1, np.inf)
          The order of the exponential. In the Vision Research sparse coding
          paper[1] this is 4. We can make the cutoff sharper by increasing
          this number.
      ** elif...
  norm_and_threshold : bool
      If true, make sure the maximum magnitude of the transfer function is 1.0
      and threshold any values below 1e-5

  Returns
  -------
  lpf_DFT = ndarray(complex128, size(DFT_num_samples[0], DFT_num_samples[1]))
      The DFT of the low pass filter
  """
  if filter_parameters['shape'] == 'exponential':
    assert all([x in filter_parameters for x in ['cutoff', 'order']])
    assert (filter_parameters['cutoff'] >= 0.0 and
            filter_parameters['cutoff'] <= 1.0)
    assert filter_parameters['order'] >= 1.0

    freqs_vert = np.fft.fftfreq(DFT_num_samples[0])
    freqs_horz = np.fft.fftfreq(DFT_num_samples[1])

    two_d_freq = np.meshgrid(freqs_vert, freqs_horz, indexing='ij')
    spatial_freq_mag = (np.sqrt(np.square(two_d_freq[0]) +
                                np.square(two_d_freq[1])))
    lpf_DFT_mag = np.exp(
        -1. * np.power(spatial_freq_mag / (0.5 * filter_parameters['cutoff']),
                       filter_parameters['order']))
    #^ 0.5 is the 2d spatial nyquist frequency
    if norm_and_threshold:
      lpf_DFT_mag[lpf_DFT_mag < 1e-3] = 1e-3
    #^ avoid filter magnitudes that are 'too small' because this will make
    #  undoing the filter introduce arbitrary high-frequency noise
    lpf_DFT_phase = np.zeros(spatial_freq_mag.shape)
  else:
    raise KeyError('Unrecognized filter shape: ' + filter_parameters['shape'])

  return lpf_DFT_mag * np.exp(1j * lpf_DFT_phase)


def get_whitening_ramp_filter(DFT_num_samples, norm_and_threshold=True):
  """
  Returns the DFT of a simple 'magnitude ramp' filter that whitens data

  Parameters
  ----------
  DFT_num_samples : (int, int)
      The number of samples in the DFT vertical ax and the DFT horizontal ax

  Returns
  -------
  wf_DFT = ndarray(complex128, size(DFT_num_samples[0], DFT_num_samples[1]))
      The DFT of the whitening filter
  """
  freqs_vert = np.fft.fftfreq(DFT_num_samples[0])
  freqs_horz = np.fft.fftfreq(DFT_num_samples[1])

  two_d_freq = np.meshgrid(freqs_vert, freqs_horz, indexing='ij')
  spatial_freq_mag = (np.sqrt(np.square(two_d_freq[0]) +
                              np.square(two_d_freq[1])))
  if norm_and_threshold:
    wf_DFT_mag = spatial_freq_mag / np.max(spatial_freq_mag)
    wf_DFT_mag[wf_DFT_mag < 1e-5] = 1e-5
    #^ avoid filter magnitudes that are 'too small' because this will make
    #  undoing the filter introduce arbitrary high-frequency noise
  else:
    wf_DFT_mag = spatial_freq_mag
  return wf_DFT_mag * np.exp(1j * 0)  # zero-phase filter


def get_gabor_filter(filter_size, filter_parameters):
  """
  Generate a spatial filter based on a Gabor function.

  Here is my convention for parameterizing a Gabor, which may be slightly
  different than what you've see online or elsewhere.
  1) There are two orthogonal axes for a Gabor, the direction
     (aligned/parallel) to the grating is the primary axis. It is also the axis
     along which the envelope is larger. The direction normal to the grating
     is the secondary axis.
  2) Orientation is in *radians counter-clockwise from horizontal* of the
     *primary* axis. At 0 radians, the filter is most sensitive to
     horizontal lines. At pi/2 it is most sensitive to vertical lines.
  3) The envelope width is the standard deviation \sigma of the guassian
     envelope along the primary axis
  4) The envelope aspect ratio is \sigma along the secondary axis divided by
     \sigma along the primary axis. It should be less than 1.

  Parameters
  ----------
  filter_size : (int, int)
      The spatial size of the filter, in pixels
  filter_parameters : dictionary
      'orientation' : float
        The orientation of the filter, in radians
      'envelope_width' : float
        The standard deviation of the gaussian envelope along the primary axis
      'envelope_aspect' : float
        The ratio of envelope size between the secondary and primary axes.
      'frequency' : float
        Frequency, in cycles / pixel.
      'phase' : float
        Phase offest in radians (sin() used for grating).
      'position_yx' : (int, int)
        The position, in pixels, of the Gabor, relative to the center of the
        filter. This uses "array-indexing" rather than "Cartesian" indexing
        (as does this entire library) so increasing y is *down*.

  Returns
  -------
  gabor : ndarray(float32, size=(filter_size[0], filter_size[1]))
      The Gabor function filter. Will have an l2-norm of 1.0

  Example
  -------
  get_gabor_filter(filter_size=(16, 16), filter_parameters={
    'orientation': np.pi/6, 'envelope_width'=4, 'envelope_aspect': 0.75,
    'frequency': 1/8, 'phase'=0, 'position_yx'=(3, 2))
  """
  assert filter_size[0] > 2 and filter_size[1] > 2
  assert (filter_parameters['envelope_width'] <= filter_size[0] and
          filter_parameters['envelope_width'] <= filter_size[1])
  assert filter_parameters['envelope_aspect'] <= 1.0
  v_coords = np.arange(-int(np.floor(filter_size[0]/2)),
                       int(np.ceil(filter_size[0]/2)))
  h_coords = np.arange(-int(np.floor(filter_size[1]/2)),
                       int(np.ceil(filter_size[1]/2)))
  assert filter_parameters['position_yx'][0] >= np.min(v_coords)
  assert filter_parameters['position_yx'][1] >= np.min(h_coords)

  mv, mh = np.meshgrid(v_coords, h_coords, indexing='ij')
  mv_trans, mh_trans = np.meshgrid(
      filter_parameters['position_yx'][0] * np.ones(filter_size[0]),
      filter_parameters['position_yx'][1] * np.ones(filter_size[1]),
      indexing='ij')
  mh_prime = ((mh-mh_trans) * np.cos(filter_parameters['orientation']) -
              (mv-mv_trans) * np.sin(filter_parameters['orientation']))
  mv_prime = ((mh-mh_trans) * np.sin(filter_parameters['orientation']) +
              (mv-mv_trans) * np.cos(filter_parameters['orientation']))
  envelope = np.exp(-1 * (
    (mh_prime**2) + ((mv_prime / filter_parameters['envelope_aspect'])**2))
    / (2*(filter_parameters['envelope_width']**2)))
  grating = np.sin(2 * np.pi * filter_parameters['frequency'] * mv_prime +
                   filter_parameters['phase'])
  gabor = envelope * grating
  return gabor / np.linalg.norm(gabor)


def whiten_center_surround(image, cutoffs, return_filter=False,
                           norm_and_threshold=True):
  """
  Slight mod to scheme described in the Vision Research sparse coding paper [1]

  We have the composition of a low pass filter with a ramp in spatial frequency
  which together produces a center-surround filter in the image domain. The
  wrinkle is that we can flatten off the transfer function in the low frequency
  band so it passes some low frequency information through. This is useful for
  making the unwhitening computation more well-behaved and for passing of the
  responsibility of coding low-freq information to a subsequent step of
  processing, for instance `local_luminance_subtraction` (see below).

  Parameters
  ----------
  image : ndarray(float32 or uint8, size=(h, w, c))
      An image of height h and width w, with c color channels
  return_filter : bool, optional
      If true, also return the DFT of the used filter. Just for unwhitening,
      visualization, and debugging purposes. Default False.
  """
  assert image.dtype in ['float32', 'uint8']
  lpf = get_low_pass_filter(image.shape,
      {'shape': 'exponential', 'cutoff': cutoffs['high'], 'order': 8.0},
      norm_and_threshold=False)
  wf = get_whitening_ramp_filter(image.shape, norm_and_threshold=False)
  rolled_off_ramp = np.maximum(wf, cutoffs['low'] * np.ones(wf.shape))
  combined_filter = rolled_off_ramp * lpf
  if norm_and_threshold:
    combined_filter /= np.max(np.abs(combined_filter))
    combined_filter[np.abs(combined_filter) < 1e-3] = 1e-3
  if return_filter:
    return filter_fd(image, combined_filter), combined_filter
  else:
    return filter_fd(image, combined_filter)


def unwhiten_center_surround(image, low_cutoff=None, orig_filter_DFT=None):
  """
  Undoes center-surround whitening. If the original filter DFT is provided
  this is used to exactly invert the filter. If not provided, we drop the
  the low pass part and whiten with a rolled-off ramp filter based on the
  specified cutoffs.

  Parameters
  ----------
  image : ndarray(float32, size=(h, w, c))
      An image of height h and width w, with c color channels
  low_cutoff : float, optional
      The lowpass cutoff that rolls off the ramp filter to pass DC. Ignored
      if orig_filter_DFT is provided. Default None.
  orig_filter_DFT : ndarray(complex128), optional
      If not None, this is used to invert the whitening exactly. Default None.
  """
  assert image.dtype == 'float32'
  assert not ((low_cutoff is None) and (orig_filter_DFT is None))
  if orig_filter_DFT is None:
    # synthesize a whitening filter but don't use the upper low pass filter.
    # inverting this causes problems with noise and instability
    wf = get_whitening_ramp_filter(image.shape, norm_and_threshold=False)
    orig_filter_DFT = np.maximum(wf, low_cutoff * np.ones(wf.shape))
  return filter_fd(image, 1. / orig_filter_DFT)


def whiten_ZCA(flat_data, precomputed_ZCA_parameters=None):
  """
  Uses the principal components transformation to whiten data.

  We have to use a large dataset to estimate the directions of largest variance
  in vector-valued data, the principal components, and then we normalize the
  variance of each direction in this space of principal components. This has
  a similar, *but different* visual affect on images than does the
  whiten_center_surround transformation.

  Parameters
  ----------
  flat_data : ndarray(float32 or uint8, size=(D, n))
      n is the dimensionality of a single datapoint and D is the size of the
      dataset to which we are applying the ZCA transform (independently to
      each sample). We may also be using this dataset to estimate the ZCA
      whitening transform in the first place.
  precomputed_ZCA_parameters : dictionary, optional
      The parameters of a ZCA transform that have already been estimated. If
      None, we will compute these based on flat_data. Default None.
      'PCA_basis' : ndarray(float32, size=(n, n))
        The principal components transform matrix meant to be applied with a
        right inner product to new data
      'PCA_axis_variances' : ndarray(float32, size=(n,))
          The estimated variance of data on each of the n princpal axes.
      'subtracted_mean' : float32
          We subtract this from each datapoint to approximately 'zero' the data

  Returns
  -------
  whitened_data : ndarray(float32, size=(D, n))
      The data, now whitened
  ZCA_parameters : dictionary, if precomputed_ZCA_parameters is None
      The parameters of a ZCA transform that we estimate from flat_data.
      'PCA_basis' : ndarray(float32, size=(n, n))
        The principal components transform matrix meant to be applied with a
        left inner product to new data
      'PCA_axis_variances' : ndarray(float32, size=(n,))
          The estimated variance of data on each of the n princpal axes.
      'subtracted_mean' : float32
          We subtract this from each datapoint to approximately 'zero' the data
  """
  assert flat_data.dtype in ['float32', 'uint8']
  # we could do all this in torch using the functionality defined in
  # ../training/pca.py and ../analysis_transforms/invertible_linear.py, but
  # I'm trying to make this portable and numpy/pythonic, so we'll do it
  # manually here
  num_samples = flat_data.shape[0]
  num_components = flat_data.shape[1]
  if precomputed_ZCA_parameters is None:
    if num_components > 0.1 * num_samples:
      raise RuntimeError('Number of samples is way too small to estimate PCA')
    meanzero_flat_data, component_means = center_each_component(flat_data)
    U, w, _ = np.linalg.svd(
        np.dot(meanzero_flat_data.T, meanzero_flat_data) / num_samples,
        full_matrices=True)
    # ^way faster to estimate based on the n x n covariance matrix
    ZCA_parameters = {'PCA_basis': U, 'PCA_axis_variances': w,
                      'subtracted_mean': np.mean(component_means)}
    # technically speaking, we should subtract from each component its
    # specific mean over the dataset. However, when we patch an image, compute
    # the transform, and then reassemble the patches, using component-specific
    # means can introduce some blocking artifacts in the reassembled image.
    # Assuming there's nothing special about particular components, they will
    # all have approximately the same mean. Instead, I will just subtract the
    # mean of these means, approximately zeroing the data while reducing the
    # visual artifacts.
  else:
    # shallow copy just creates a reference, not an honest-to-goodness copy
    ZCA_parameters = precomputed_ZCA_parameters.copy()
    meanzero_flat_data = flat_data - ZCA_parameters['subtracted_mean']

  meanzero_white_data = np.dot(
      np.dot(meanzero_flat_data, ZCA_parameters['PCA_basis']) /
      (np.sqrt(ZCA_parameters['PCA_axis_variances']) + 1e-4)[None, :],
      ZCA_parameters['PCA_basis'].T)

  white_data = (meanzero_white_data.astype('float32') +
                ZCA_parameters['subtracted_mean'])

  if precomputed_ZCA_parameters is None:
    return white_data, ZCA_parameters
  else:
    return white_data


def unwhiten_ZCA(white_flat_data, precomputed_ZCA_parameters):
  """
  Undoes the ZCA whitening operation (see above)

  Parameters
  ----------
  white_flat_data : ndarray(float32, size=(D, n))
      n is the dimensionality of a single datapoint and D is the size of the
      dataset to which we are applying the ZCA transform (independently to
      each sample).
  precomputed_ZCA_parameters : dictionary
      The parameters of a ZCA transform that have already been estimated.
      'PCA_basis' : ndarray(float32, size=(n, n))
        The principal components transform matrix meant to be applied with a
        right inner product to new flat data
      'PCA_axis_variances' : ndarray(float32, size=(n,))
          The estimated variance of data on each of the n princpal axes.
      'subtracted_mean' : float32
          We subtract this from each datapoint to approximately 'zero' the data

  Returns
  -------
  colored_data : ndarray(float32, size=(D, n))
      The data, with whitening operation undone
  """
  assert white_flat_data.dtype == 'float32'
  meanzero_white_data = (white_flat_data -
                         precomputed_ZCA_parameters['subtracted_mean'])
  meanzero_colored_data = np.dot(
      np.dot(meanzero_white_data, precomputed_ZCA_parameters['PCA_basis']) *
      (np.sqrt(precomputed_ZCA_parameters['PCA_axis_variances']) + 1e-4)[None, :],
      precomputed_ZCA_parameters['PCA_basis'].T)

  colored_data = (meanzero_colored_data.astype('float32') +
                  precomputed_ZCA_parameters['subtracted_mean'])

  return colored_data


def local_contrast_normalization(image, filter_sigma, return_normalizer=False):
  """
  Computes an estimate of the local contrast and removes this from an image

  Parameters
  ----------
  image : ndarray(float32 or uint8, size=(h, w, c))
      An image of height h and width w, with c color channels
  filter_sigma : tuple(int, int)
      The standard deviation of the isotropic gaussian kernel that we use to
      compute a local estimate of the variance
  return_normalizer : bool, optional
      If true, return the array used to do the normalization -- this can be
      used to reverse the transform. Defualt False.

  Returns
  -------
  filtered_image : ndarray(float32, size=(h, w, c))
  normalizer : ndarray(float32, size=(h, w, c)), if return_normalizer=True
  """
  window_size = 2  # in terms of sigma. This gives the size of FIR filter
  v_coords = np.arange(-int(np.ceil(window_size*filter_sigma)),
                       int(np.ceil(window_size*filter_sigma))+1)
  h_coords = np.arange(-int(np.ceil(window_size*filter_sigma)),
                       int(np.ceil(window_size*filter_sigma))+1)
  kv_coords, kh_coords = np.meshgrid(v_coords, h_coords, indexing='ij')
  gaussian_kernel = np.exp(-0.5*(kv_coords**2 + kh_coords**2) /
                           (filter_sigma**2))
  gaussian_kernel /= np.sum(gaussian_kernel)

  local_variance = filter_sd(image**2, gaussian_kernel)

  # TODO: deal with divide by zero
  local_variance[local_variance == 0] = 1.
  if return_normalizer:
    return image / np.sqrt(local_variance), np.sqrt(local_variance)
  else:
    return image / np.sqrt(local_variance)


def local_luminance_subtraction(image, filter_sigma, return_subtractor=False):
  """
  Computes an estimate of the local luminance and removes this from an image

  Parameters
  ----------
  image : ndarray(float32 or uint8, size=(h, w, c))
      An image of height h and width w, with c color channels
  filter_sigma : float
      The standard deviation of the isotropic gaussian kernel that we use to
      compute a local estimate of the luminance
  return_subtractor : bool, optional
      If true, return the array used to do the luminance subtraction -- this
      can be used to reverse the transform. Defualt False.

  Returns
  -------
  filtered_image : ndarray(float32, size=(h, w, c))
  subtractor : ndarray(float32, size=(h, w, c))
  """
  window_size = 2  # in terms of sigma. This gives the size of FIR filter
  v_coords = np.arange(-int(np.ceil(window_size*filter_sigma)),
                       int(np.ceil(window_size*filter_sigma))+1)
  h_coords = np.arange(-int(np.ceil(window_size*filter_sigma)),
                       int(np.ceil(window_size*filter_sigma))+1)
  kv_coords, kh_coords = np.meshgrid(v_coords, h_coords, indexing='ij')
  gaussian_kernel = np.exp(-0.5*(kv_coords**2 + kh_coords**2) /
                           (filter_sigma**2))
  gaussian_kernel /= np.sum(gaussian_kernel)

  local_luminance = filter_sd(image, gaussian_kernel)

  if return_subtractor:
    return image - local_luminance, local_luminance
  else:
    return image - local_luminance


def center_each_component(flat_data):
  """
  Makes each component of data have mean zero across the dataset

  Parameters
  ----------
  flat_data : ndarray(float32 or uint8, size=(D, n))
      n is the dimensionality of a single datapoint and D is the size of the
      dataset over which we are taking the mean.

  Returns
  -------
  centered_data : ndarray(float32, size=(D, n))
      The data, now with mean 0 in each component
  original_means : ndarray(float32, size=(n,))
      The componentwise means of the original data. Can be used to
      uncenter the data later (for instance, after dictionary learning)
  """
  assert flat_data.dtype in ['float32', 'uint8']
  original_means = np.mean(flat_data, axis=0)
  return (flat_data - original_means[None, :]).astype('float32'), original_means


def center_each_sample(flat_data):
  """
  Makes each sample have an average value of zero

  Parameters
  ----------
  flat_data : ndarray(float32 or uint8, size=(D, n))
      n is the dimensionality of a single datapoint and D is the size of the
      dataset over which we are taking the mean.

  Returns
  -------
  zero_dc_data : ndarray(float32, size=(D, n))
      The data, now with a DC value of zero
  original_means : ndarray(float32, size=(n,))
      The patch-specific DC values of the original data. Can be used to
      uncenter the data later (for instance, after dictionary learning)
  """
  assert flat_data.dtype in ['float32', 'uint8']
  original_means = np.mean(flat_data, axis=1)
  return (flat_data - original_means[:, None]).astype('float32'), original_means


def normalize_component_variance(flat_data):
  """
  Normalize each component to have a variance of 1 across the dataset

  Parameters
  ----------
  flat_data : ndarray(float32 or uint8, size=(D, n))
      n is the dimensionality of a single datapoint and D is the size of the
      dataset over which we are taking the variance.

  Returns
  -------
  normalized_data : ndarray(float32 size=(D, n))
      The data, now with variance
  original_variances : ndarray(float32, size=(n,))
      The componentwise variances of the original data. Can be used to
      unnormalize the data later (for instance, after dictionary learning)
  """
  assert flat_data.dtype in ['float32', 'uint8']
  original_variances = np.var(flat_data, axis=0)
  return ((flat_data / np.sqrt(original_variances)[None, :]).astype('float32'),
          original_variances)


def patches_from_single_image(image, patch_dimensions, flatten_patches):
  """
  Extracts tiled patches from a single image

  Parameters
  ----------
  image : ndarray(float32 or uint8, size=(h, w, c))
      An image of height h and width w, with c color channels
  patch_dimensions : tuple(int, int)
      The size in pixels of each patch
  flatten_patches : bool
      Indicates whether to flatten the patches or leave them in
      (ph, pw, c) format.

  Returns
  -------
  patches : ndarray(float32, size=(k, ph*pw*c) OR (k, ph, pw, c))
      An array of patches each of height ph and width pw. k is the
      number of total patches that were extracted from the full image
  patch_positions : list(tuple(int, int))
      The position in pixels of the upper-left-hand corner of each patch within
      the full image. Used to re-tile the full image after processing
      the patches
  """
  assert image.dtype in ['float32', 'uint8']
  assert image.ndim == 3
  if (image.shape[0] / patch_dimensions[0] % 1 != 0 or
      image.shape[1] / patch_dimensions[1] % 1 != 0):
    print('Warning: image cannot be completely patched with these dimensions.',
          'Ignoring overflow pixels on the right and bottom of image')

  num_patches_vert = image.shape[0] // patch_dimensions[0]
  num_patches_horz = image.shape[1] // patch_dimensions[1]
  patches = np.zeros([num_patches_vert * num_patches_horz,
                      patch_dimensions[0], patch_dimensions[1],
                      image.shape[2]], dtype=image.dtype)
  patch_positions = []  # keep track of where each patch belongs
  p_idx = 0
  for patch_idx_vert in range(num_patches_vert):
    for patch_idx_horz in range(num_patches_horz):
      pos_vert = patch_idx_vert * patch_dimensions[0]
      pos_horz = patch_idx_horz * patch_dimensions[1]
      patches[p_idx] = image[pos_vert:pos_vert+patch_dimensions[0],
                             pos_horz:pos_horz+patch_dimensions[1]]
      patch_positions.append((pos_vert, pos_horz))
      #^ upper left hand corner position in pixels in the original image
      p_idx += 1

  if flatten_patches:
    patches = patches.reshape((patches.shape[0], -1))

  return patches, patch_positions


def assemble_image_from_patches(patches, patch_dimensions, patch_positions):
  """
  Tiles an image from patches

  Parameters
  ----------
  patches : ndarray(float32 or uint8, size=(k, ph*pw*c) OR (k, ph, pw, c))
      An array of patches each of height ph and width pw. k is the
      number of total patches that were extracted from the full image
  patch_dimensions : tuple(int, int)
      The size in pixels of each patch
  patch_positions : list(tuple(int, int))
      The position in pixels of the upper-left-hand corner of each patch within
      the full image.

  Returns
  -------
  image : ndarray(float32 or uint8, size=(h, w, c))
      An image of height h and width w
  """
  assert patches.dtype in ['float32', 'uint8']

  full_img_height = (np.max([x[0] for x in patch_positions]) +
                     patch_dimensions[0])
  full_img_width = (np.max([x[1] for x in patch_positions]) +
                    patch_dimensions[1])
  if patches.ndim == 2:
    # flattened patches
    num_color_channels = (patches.shape[1] /
                          (patch_dimensions[0]*patch_dimensions[1]))
    assert num_color_channels % 1.0 == 0
    num_color_channels = int(num_color_channels)
  else:
    num_color_channels = patches.shape[-1]

  full_img = np.zeros([full_img_height, full_img_width,
                       num_color_channels], dtype=patches.dtype)
  for patch_idx in range(patches.shape[0]):
    vert = patch_positions[patch_idx][0]
    horz = patch_positions[patch_idx][1]
    if patches.ndim == 2:
      full_img[vert:vert+patch_dimensions[0], horz:horz+patch_dimensions[1]] = \
          patches[patch_idx, :].reshape(
              (patch_dimensions[0], patch_dimensions[1], num_color_channels))
    else:
      full_img[vert:vert+patch_dimensions[0], horz:horz+patch_dimensions[1]] = \
          patches[patch_idx, :]

  return full_img
