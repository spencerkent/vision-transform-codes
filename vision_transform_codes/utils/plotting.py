"""
Some simple utilities for plotting our transform codes
"""

import bisect
import numpy as np
from scipy.stats import kurtosis
from skimage.measure import compare_ssim as ssim
from matplotlib import pyplot as plt
from matplotlib.image import NonUniformImage
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter

tab10colors = plt.get_cmap('tab10').colors
blue_red = plt.get_cmap('RdBu_r')


def compute_pSNR(target, reconstruction, manual_sig_mag=None):
  if manual_sig_mag is None:
    signal_magnitude = np.max(target) - np.min(target)
  else:
    signal_magnitude = manual_sig_mag

  MSE = np.mean(np.square(target - reconstruction))
  if MSE != 0:
    return 10. * np.log10((signal_magnitude**2)/MSE)
  else:
    return np.inf


def compute_ssim(target, reconstruction, manual_sig_mag=None):
  if manual_sig_mag is None:
    signal_magnitude = np.max(target) - np.min(target)
  else:
    signal_magnitude = manual_sig_mag
  # these are the settings that the scikit-image documentation indicates
  # match the ones chosen in the original SSIM paper (Wang 2004, I believe).
  return ssim(target, reconstruction, data_range=signal_magnitude,
              gaussian_weights=True, sigma=1.5, use_sample_covariance=False)


def standardize_for_imshow(image):
  """
  A luminance standardization for pyplot's imshow

  This just allows me to specify a simple, transparent standard for what white
  and black correspond to in pyplot's imshow method. Likely could be
  accomplished by the colors.Normalize method, but I want to make this as
  explicit as possible. If the image is nonnegative, we divide by the scalar
  that makes the largest value 1.0. If the image is nonpositive, we
  divide by the scalar that makes the smallest value -1.0, and then add 1, so
  that this value is 0.0, pitch black. If the image has both positive and
  negative values, we divide and shift so that 0.0 in the original image gets
  mapped to 0.5 for imshow and the largest absolute value gets mapped to
  either 0.0 or 1.0 depending on whether it was positive of negative.

  Parameters
  ----------
  image : ndarray
      The image to be standardized, can be (h, w) or (h, w, c). All operations
      are scalar operations applied to every color channel.

  Returns
  -------
  standardized_image : ndarray
      An RGB image in the range [0.0, 1.0], ready to be showed by imshow.
  raw_val_mapping : tuple(float, float, float)
      Indicates what raw values got mapped to 0.0, 0.5, and 1.0, respectively
  """
  max_val = np.max(image)
  min_val = np.min(image)
  if max_val == min_val:  # constant value
    standardized_image = 0.5 * np.ones(image.shape)
    if max_val > 0:
      raw_val_mapping = [0.0, max_val, 2*max_val]
    elif max_val < 0:
      raw_val_mapping = [2*max_val, max_val, 0.0]
    else:
      raw_val_mapping = [-1.0, 0.0, 1.0]
  else:
    if min_val >= 0:
      standardized_image = image / max_val
      raw_val_mapping = [0.0, 0.5*max_val, max_val]
    elif max_val <= 0:
      standardized_image = (image / -min_val) + 1.0
      raw_val_mapping = [min_val, 0.5*min_val, 0.0]
    else:
      # straddles 0.0. We want to map 0.0 to 0.5 in the displayed image
      skew_toward_max = np.argmax([abs(min_val), abs(max_val)])
      if skew_toward_max:
        normalizer = (2 * max_val)
        raw_val_mapping = [-max_val, 0.0, max_val]
      else:
        normalizer = (2 * np.abs(min_val))
        raw_val_mapping = [min_val, 0.0, -min_val]
      standardized_image = (image / normalizer) + 0.5
  return standardized_image, raw_val_mapping


def display_dictionary(dictionary, renormalize=False, reshaping=None,
                       highlighting=None, plot_title=""):
  """
  Plot each of the dictionary elements side by side

  Parameters
  ----------
  dictionary : ndarray(float32, size=(s, n) OR (s, c, kh, kw))
      If the size of dictionary is (s, n), this is a 'fully-connected'
      dictionary where each basis element has the same dimensionality as the
      image it is trying to represent. n is the size of the image and s the
      number of basis functions. If the size of dictionary is (s, c, kh, kw),
      this is a 'convolutional' dictionary where each basis element is
      (potentially much) smaller than the image it is trying to represent. c
      is the number of channels that in the input space, kh is the dictionary
      kernel height, and kw is the dictionary kernel width.
  renormalize : bool, optional
      If present, display basis functions on their own color scale, using
      standardize_for_imshow() to put values in the range [0, 1]. Will
      accentuate the largest-magnitude values in the dictionary element.
      Default False.
  reshaping : tuple(int, int), optional
      Should only be specified for a fully-connected dictionary (where
      dictionary.ndim==2). The dimension of each patch before vectorization
      to size n. We reshape the dictionary elements based on this. Default None
  highlighting : dictionary, optional
      This is used to re-sort and color code the dictionary elements according
      to scalar weights. Has two keys:
      'weights' : ndarray(float, size=(s,))
        The weights for each dictionary element
      'color_range': tuple(float, float)
        Values less than or equal to highlighting['color_range'][0] get mapped
        to dark blue, and values greater than or equal to
        highlighting['color_range'][1] get mapped to dark red.
      Default None.
  plot_title : str, optional
      The title of the plot. Default ""

  Returns
  -------
  dictionary_figs : list
      A list containing pyplot figures. Can be saved separately, or whatever
      from the calling function
  """
  t_ims, raw_val_mapping = get_dictionary_tile_imgs(
      dictionary, reshape_to_these_dims=reshaping, indv_renorm=renormalize,
      highlights=highlighting)
  fig_refs = []
  for fig_idx in range(len(t_ims)):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes([0.075, 0.075, 0.85, 0.85])  # [bottom, left, height, width]
    fig.suptitle(plot_title + ', fig {} of {}'.format(
                 fig_idx+1, len(t_ims)), fontsize=20)
    im_ref = ax.imshow(t_ims[fig_idx], interpolation='None')
    ax.axis('off')
    if not renormalize:
      # add a luminance colorbar. Because there isn't good rgb colorbar
      # support in pyplot I hack this by adding another image subplot
      cbar_ax = plt.axes([0.945, 0.4, 0.01, 0.2])
      gradient = np.linspace(1.0, 0.0, 256)[:, None]
      cbar_ax.imshow(gradient, cmap='gray')
      cbar_ax.set_aspect('auto')
      cbar_ax.yaxis.tick_right()
      cbar_ax.xaxis.set_ticks([])
      cbar_ax.yaxis.set_ticks([255, 128, 0])
      cbar_ax.yaxis.set_ticklabels(['{:.2f}'.format(x)
                                    for x in raw_val_mapping], fontsize=8)
    fig_refs.append(fig)

  return fig_refs


def get_dictionary_tile_imgs(dictionary, indv_renorm=False,
                             reshape_to_these_dims=None, highlights=None):
  """
  Arranges a dictionary into a series of imgs that tile elements side by side

  We do some simple rescaling to provide a standard interpretation of white and
  black pixels in the image (and everything in-between).

  Parameters
  ----------
  dictionary : ndarray(float32, size=(s, n) OR (s, c, kh, kw))
      See docstring of display_dictionary above.
  indv_renorm : bool, optional
      See docstring of display_dictionary above.
  reshape_to_these_dims : tuple(int, int), optional
      See docstring of display_dictionary above.
  highlights : dictionary, optional
      See docstring of display_dictionary above.

  Returns
  -------
  tile_imgs : list(ndarray)
      Each element is an image to be displayed by imshow
  imshow_to_raw_mapping : tuple(float, float, float)
      Returned by standardize_for_imshow(), this indicates which values in the
      original dictionary got mapped to 0.0, 0.5, and 1.0, respectively, in
      the displayed image.
  """
  if indv_renorm:
    imshow_to_raw_mapping = None  # each dict element put on their own scale
  else:
    dictionary, imshow_to_raw_mapping = standardize_for_imshow(dictionary)

  if highlights is not None:
    # reorder by weight
    new_ordering = np.argsort(highlights['weights'])[::-1]
    dictionary = dictionary[new_ordering]
    highlights['weights'] = highlights['weights'][new_ordering]
    weight_colors = (
        (highlights['weights'] - highlights['color_range'][0]) /
        (highlights['color_range'][1] - highlights['color_range'][0]))
    if highlights['color_range'][0] >= 0 or highlights['color_range'][1] <= 0:
      print('Warning: Red and Blue will not correspond',
            'to positive and negative weights')

  max_de_per_img = 80*80  # max 80x80 {d}ictionary {e}lements per tile img
  assert np.sqrt(max_de_per_img) % 1 == 0, 'please pick a square number'
  num_de = dictionary.shape[0]
  num_tile_imgs = int(np.ceil(num_de / max_de_per_img))
  # this determines how many dictionary elements are arranged in a square
  # grid within any given img
  if num_tile_imgs > 1:
    de_per_img = max_de_per_img
  else:
    squares = [x**2 for x in range(1, int(np.sqrt(max_de_per_img))+1)]
    de_per_img = squares[bisect.bisect_left(squares, num_de)]
  plot_sidelength = int(np.sqrt(de_per_img))

  if dictionary.ndim == 2:
    assert reshape_to_these_dims is not None
    basis_elem_size = reshape_to_these_dims
  else:
    basis_elem_size = np.array(dictionary.shape[1:])[[1, 2, 0]]

  if highlights is None:
    h_margin = 2
    w_margin = 2
  else:
    # little extra room for the highlights
    h_margin = 6
    w_margin = 6
    hl_h_margin = 2  # pixel width of the highlights
    hl_w_margin = 2
  full_img_height = (basis_elem_size[0] * plot_sidelength +
                     (plot_sidelength + 1) * h_margin)
  full_img_width = (basis_elem_size[1] * plot_sidelength +
                    (plot_sidelength + 1) * w_margin)

  de_idx = 0
  tile_imgs = []
  for in_de_img_idx in range(num_tile_imgs):
    image_matrix_shape = (full_img_height, full_img_width, 3)
    composite_img = np.ones(image_matrix_shape)
    img_de_idx = de_idx % de_per_img
    while img_de_idx < de_per_img and de_idx < num_de:

      if dictionary.ndim == 2:
        this_de = dictionary[de_idx, :]
        if len(basis_elem_size) == 2:
          this_de = np.repeat(
              this_de.reshape(basis_elem_size)[:, :, None], 3, axis=2)
        else:
          this_de = this_de.reshape(basis_elem_size)
      else:
        this_de = np.moveaxis(dictionary[de_idx], 0, 2)

      if indv_renorm:
        this_de, _ = standardize_for_imshow(this_de)

      # okay, now actually place the DEs in this tile image
      row_idx = img_de_idx // plot_sidelength
      col_idx = img_de_idx % plot_sidelength
      pxr1 = row_idx * (basis_elem_size[0] + h_margin) + h_margin
      pxr2 = pxr1 + basis_elem_size[0]
      pxc1 = col_idx * (basis_elem_size[1] + w_margin) + w_margin
      pxc2 = pxc1 + basis_elem_size[1]
      composite_img[pxr1:pxr2, pxc1:pxc2] = this_de
      if highlights is not None:
        rgb_color = blue_red(weight_colors[de_idx])[:3]
        composite_img[pxr1-hl_h_margin:pxr1,
                      pxc1-hl_w_margin:pxc2 + hl_w_margin] = rgb_color
        composite_img[pxr2:pxr2+hl_h_margin,
                      pxc1-hl_w_margin:pxc2 + hl_w_margin] = rgb_color
        composite_img[pxr1-hl_h_margin:pxr2 + hl_h_margin,
                      pxc1-hl_w_margin:pxc1] = rgb_color
        composite_img[pxr1-hl_h_margin:pxr2 + hl_h_margin,
                      pxc2:pxc2+hl_w_margin] = rgb_color

      img_de_idx += 1
      de_idx += 1

    tile_imgs.append(composite_img)

  return tile_imgs, imshow_to_raw_mapping


def display_codes(codes, indv_stem_plots=True,
                  input_and_recon=None, plot_title=""):
  """
  Visualizes tranform codes

  Parameters
  ----------
  codes : ndarray(float32, size=(b, s) OR size=(b, s, sh, sw)
      The codes for a batch of size b. b shouldn't be too large unless you want
      to wait around all day for this to finish. If this is 4-dimensional
      it means that there is some 2d layout to each code. Each code element
      can be interpreted as a channel in an image
  indv_stem_plots : bool, optional
      Use an individual stem plot for each code. The alternative is to just
      pack these into an image and display in greyscale. Default True
  input_and_recon : dictionary
      Input and reconstruction pairs to display alongside the code, to give a
      sense for what the code represents, and how well it does this.
      'input' : ndarray(float32, size=(b, f) OR (b, h, w), OR (b, h, w, c))
        The input. Just display as-is -- could be flat but most likely is an
        image patch.
      'recon' : ndarray(same dims as 'input')
        The corresponding reconstruction from the transform code
      'vrange' : the imshow value range on which to display these.
  plot_title : str, optional
      The title of the plot. Default ""

  Returns
  -------
  code_figs : list
      A list containing pyplot figures. Can be saved separately, or whatever
      from the calling function
  """
  # TODO: get this set up for convolutional codes
  num_codes = codes.shape[0]
  code_size = codes.shape[1]
  if indv_stem_plots:
    max_data_pt_per_fig = 25
    code_inds = np.arange(code_size)
  else:
    max_data_pt_per_fig = 1000
  num_code_figs = int(np.ceil(num_codes / max_data_pt_per_fig))
  if num_code_figs > 1:
    data_pt_per_fig = max_data_pt_per_fig
  else:
    data_pt_per_fig = num_codes

  dpt_idx = 0
  code_figs = []
  for in_code_fig_idx in range(num_code_figs):
    if indv_stem_plots:
      fig = plt.figure(figsize=(10, 10))
      if input_and_recon is not None:
        gridspec = fig.add_gridspec(ncols=3, nrows=data_pt_per_fig,
            width_ratios=[20, 1, 1], height_ratios=[1]*data_pt_per_fig,
            hspace=0.1, wspace=0.1)
      else:
        gridspec = fig.add_gridspec(ncols=1, nrows=data_pt_per_fig,
            width_ratios=[1], height_ratios=[1]*data_pt_per_fig, hspace=1)
    else:
      fig = plt.figure(figsize=(12, 10))
      code_img = np.zeros([data_pt_per_fig, code_size])
      code_img_normalized = np.zeros([data_pt_per_fig, code_size])
    fig.suptitle(plot_title + ', fig {} of {}'.format(
                 in_code_fig_idx+1, num_code_figs), fontsize=15)
    fig_dpt_idx = dpt_idx % data_pt_per_fig
    while fig_dpt_idx < data_pt_per_fig and dpt_idx < num_codes:
      if indv_stem_plots:
        if dpt_idx % 50 == 0:
          print('plotted', dpt_idx, 'of', num_codes, 'codes')
        ax = fig.add_subplot(gridspec[fig_dpt_idx, 0])
        markerlines, stemlines, baseline = ax.stem(code_inds,
            codes[dpt_idx, :], linefmt='-', markerfmt=' ',
            use_line_collection=True)
        plt.setp(stemlines, 'color', tab10colors[0])
        plt.setp(baseline, 'color', 'k')
        plt.setp(baseline, 'linewidth', 0.25)
        ax.text(0.01, 0.75, 'L0: {:.2f}, L1: {:.1f}'.format(
          np.sum(codes[dpt_idx, :] != 0) / code_size,
          np.sum(np.abs(codes[dpt_idx, :]))), fontsize=6,
          transform=ax.transAxes, color='g')
        if fig_dpt_idx < data_pt_per_fig - 1:
          ax.set_xticks([])
        else:
          ax.set_xticks([code_size//2, code_size-1])
        ax.tick_params(axis='y', which='major', labelsize=5)
        ax.tick_params(axis='y', which='minor', labelsize=4)
        ax.yaxis.get_offset_text().set_size(5)
        ax.tick_params(axis='x', which='major', labelsize=12)
        ax.tick_params(axis='x', which='minor', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim([-2, code_size+1])
        if input_and_recon is not None:
          ax = fig.add_subplot(gridspec[fig_dpt_idx, 1])
          ax.imshow(input_and_recon['input'][dpt_idx], cmap='gray',
                    vmin=input_and_recon['vrange'][0],
                    vmax=input_and_recon['vrange'][1])
          ax.axis('off')
          if fig_dpt_idx == 0:
            ax.set_title('In', fontsize=8)
          ax = fig.add_subplot(gridspec[fig_dpt_idx, 2])
          ax.imshow(input_and_recon['recon'][dpt_idx], cmap='gray',
                    vmin=input_and_recon['vrange'][0],
                    vmax=input_and_recon['vrange'][1])
          ax.text(1.0, 0.1, '{:.1f}dB'.format(compute_pSNR(
            input_and_recon['input'][dpt_idx],
            input_and_recon['recon'][dpt_idx],
            manual_sig_mag=input_and_recon['vrange'][1]-
                           input_and_recon['vrange'][0])),
            color='w', fontsize=5, transform=ax.transAxes,
            horizontalalignment='right')
          ax.axis('off')
          if fig_dpt_idx == 0:
            ax.set_title('Rec', fontsize=8)
      else:
        code_img[fig_dpt_idx] = codes[dpt_idx]
        max_code_mag = np.max(np.abs(codes[dpt_idx]))
        if max_code_mag != 0:
          code_img_normalized[fig_dpt_idx] = codes[dpt_idx] / max_code_mag

      fig_dpt_idx += 1
      dpt_idx += 1
    if not indv_stem_plots:
      ax = fig.add_subplot(1, 2, 1)
      ax.imshow(code_img, cmap='gray', interpolation='None')
      ax.set_aspect('auto')
      ax = fig.add_subplot(1, 2, 2)
      ax.imshow(code_img_normalized, cmap='gray', interpolation='None')
      ax.set_aspect('auto')
    code_figs.append(fig)

  return code_figs


def display_code_marginal_densities(codes, num_hist_bins, log_prob=False,
    ignore_vals=[], lines=True, overlaid=False, plot_title=""):
  """
  Estimates the marginal density of coefficients of a code over some dataset

  Parameters
  ----------
  codes : ndarray(float32, size=(D, s))
      The codes for a dataset of size D. These are the vectors x for each
      sample from the dataset. The value s is the dimensionality of the code
  num_hist_bins : int
      The number of bins to use when we make a histogram estimate of the
      empirical density.
  log_prob : bool, optional
      Display probabilities on a logarithmic scale. Useful for most sparse
      codes. Default False.
  ignore_vals : list, optional
      A list of code values to ignore from the estimate. Default []. TODO:
      make this more flexible so this can ignore values in a certain range.
  lines : bool, optional
      If true, plot the binned counts using a line rather than bars. This
      can make it a lot easier to compare multiple datasets at once but
      can look kind of jagged if there aren't many samples
  overlaid : bool, optional
      If true, then make a single plot with the marginal densities all overlaid
      on top of eachother. This gets messy for more than a few coefficients.
      Alteratively, display the densities in their own separate plots.
      Default False.
  plot_title : str, optional
      The title of the plot. Default ""

  Returns
  -------
  code_density_figs : list
      A list containing pyplot figures. Can be saved separately, or whatever
      from the calling function
  """
  def filter_code_vals(scalar_code_vals):
    if len(ignore_vals) > 0:
      keep_these_inds = scalar_code_vals != ignore_vals[0]
      for i in range(1, len(ignore_vals)):
        keep_these_inds = np.logical_and(keep_these_inds,
                                         scalar_code_vals != ignore_vals[i])
      return scalar_code_vals[keep_these_inds]
    else:
      return scalar_code_vals

  # TODO: get this going for convolutional codes
  if overlaid:
    # there's just a single plot
    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(plot_title, fontsize=15)
    ax = plt.subplot(1, 1, 1)
    blue=plt.get_cmap('Blues')
    cmap_indeces = np.linspace(0.25, 1.0, codes.shape[1])

    histogram_min = np.min(codes)
    histogram_max = np.max(codes)
    histogram_bin_edges = np.linspace(histogram_min, histogram_max,
                                      num_hist_bins + 1)
    histogram_bin_centers = (histogram_bin_edges[:-1] +
                             histogram_bin_edges[1:]) / 2
    for de_idx in range(codes.shape[1]):
      code = filter_code_vals(codes[:, de_idx])
      counts, _ = np.histogram(code, histogram_bin_edges)
      empirical_density = counts / np.sum(counts)
      if lines:
        ax.plot(histogram_bin_centers, empirical_density,
                color=blue(cmap_indeces[de_idx]), linewidth=2,
                label='Coeff idx ' + str(de_idx))
      else:
        ax.bar(histogram_bin_centers, empirical_density, align='center',
               color=blue(cmap_indeces[de_idx]),
               width=histogram_bin_centers[1]-histogram_bin_centers[0],
               alpha=0.4, label='Coeff idx ' + str(de_idx))
    ax.legend(fontsize=10)
    if log_prob:
      ax.set_yscale('log')
    de_figs = [fig]

  else:
    # every coefficient gets its own subplot
    max_de_per_fig = 20*20  # max 20x20 {d}ictionary {e}lements displayed
    assert np.sqrt(max_de_per_fig) % 1 == 0, 'please pick a square number'
    num_de = codes.shape[1]
    num_de_figs = int(np.ceil(num_de / max_de_per_fig))
    # this determines how many dictionary elements are aranged in a square
    # grid within any given figure
    if num_de_figs > 1:
      de_per_fig = max_de_per_fig
    else:
      squares = [x**2 for x in range(1, int(np.sqrt(max_de_per_fig))+1)]
      de_per_fig = squares[bisect.bisect_left(squares, num_de)]
    plot_sidelength = int(np.sqrt(de_per_fig))

    de_idx = 0
    de_figs = []
    for in_de_fig_idx in range(num_de_figs):
      fig = plt.figure(figsize=(15, 15))
      fig.suptitle(plot_title + ', fig {} of {}'.format(
                   in_de_fig_idx+1, num_de_figs), fontsize=15)
      subplot_grid = gridspec.GridSpec(plot_sidelength, plot_sidelength,
                                       wspace=0.35, hspace=0.35)

      fig_de_idx = de_idx % de_per_fig
      while fig_de_idx < de_per_fig and de_idx < num_de:
        if de_idx % 100 == 0:
          print('plotted', de_idx, 'of', num_de, 'code coefficients')
        ax = plt.Subplot(fig, subplot_grid[fig_de_idx])
        code = filter_code_vals(codes[:, de_idx])
        histogram_min = min(code)
        histogram_max = max(code)
        histogram_bin_edges = np.linspace(histogram_min, histogram_max,
                                          num_hist_bins + 1)
        histogram_bin_centers = (histogram_bin_edges[:-1] +
                                 histogram_bin_edges[1:]) / 2
        counts, _ = np.histogram(code, histogram_bin_edges)
        empirical_density = counts / np.sum(counts)
        max_density = np.max(empirical_density)
        variance = np.var(code)
        hist_kurtosis = kurtosis(empirical_density, fisher=False)

        if lines:
          ax.plot(histogram_bin_centers, empirical_density,
                  color='k', linewidth=1)
        else:
          ax.bar(histogram_bin_centers, empirical_density,
                 align='center', color='k',
                 width=histogram_bin_centers[1]-histogram_bin_centers[0])

        ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
        ax.tick_params(axis='both', which='major',
                       labelsize=5)
        if histogram_min < 0.:
          ax.set_xticks([histogram_min, 0., histogram_max])
        else:
          ax.set_xticks([histogram_min, histogram_max])
        ax.text(0.1, 0.75, 'K: {:.1f}'.format(
          hist_kurtosis), transform=ax.transAxes,
          color='g', fontsize=5)
        ax.text(0.95, 0.75, 'V: {:.1f}'.format(
          variance), transform=ax.transAxes,
          color='b', fontsize=5, horizontalalignment='right')
        ax.set_yticks([0., max_density])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if log_prob:
          ax.set_yscale('log')

        fig.add_subplot(ax)
        fig_de_idx += 1
        de_idx += 1
      de_figs.append(fig)

  return de_figs


def display_2d_code_densities(two_codes, num_hist_bins,
    log_prob=False, with_contours=True, ignore_vals=[], plot_title=""):
  """
  Displays the empirical joint probability density between two elements

  Parameters
  ----------
  two_codes : ndarray(size=(D, 2))
      D samples of two code elements.
  num_hist_bins : int
      The number of histogram bins used for the estimate
  log_prob : bool, optional
      Display with log transformation to the probability. Makes is easier to
      see if there is a high probability mode dominating the values. Will be
      renormalized so that 0.0 is the lowest probability and 1.0 is the
      highest probability. Default False.
  with_contours : bool, optional
      Overlay equal-probability contours using pyplot's contour() method.
      Default True.
  ignore_vals : list, optional
      A list of code values to ignore from the estimate. Default []. TODO:
      make this more flexible so this can ignore values in a certain range.
  plot_title : str, optional
      An identifying title for this plot. Default ""

  Returns
  ----------
  joint_2d_fig : pyplot.Figure
      A figure that can be saved or whatever in the calling scope
  """
  #TODO: get this working with convolutional codes
  def filter_code_vals(td_code_vals):
    if len(ignore_vals) > 0:
      keep_these_inds = np.logical_and(td_code_vals[:, 0] != ignore_vals[0],
                                       td_code_vals[:, 1] != ignore_vals[0])
      for i in range(1, len(ignore_vals)):
        keep_these_inds = np.logical_and(keep_these_inds,
            np.logical_and(td_code_vals[:, 0] != ignore_vals[i],
                           td_code_vals[:, 1] != ignore_vals[i]))
      return td_code_vals[keep_these_inds]
    else:
      return td_code_vals

  def gen_gaussian_levels(min_l, max_l, num_l, log_probability=False):
    """Generates levels that are equally-spaced under a 2d gaussian"""
    if not log_probability:
      temp = np.geomspace(1, 2, num=num_l)  # log spacing
    else:
      temp = np.linspace(1, 2, num=num_l)
    temp /= np.sqrt(temp)  # now will be even in the square
    temp -= np.min(temp)
    temp *= ((max_l-min_l) / np.max(temp)) # make the range
    temp += min_l
    return temp

  assert two_codes.shape[1] == 2  # for now just visualize 2d RVs
  blues = plt.get_cmap('Blues')
  fig = plt.figure(figsize=(10, 10))
  fig.suptitle(plot_title, fontsize=14)
  two_codes = filter_code_vals(two_codes)
  empirical_density, x_bin_edges, y_bin_edges = np.histogram2d(
      x=two_codes[:, 0], y=two_codes[:, 1], bins=num_hist_bins,
      density=True)
  empirical_density = empirical_density.T  # indexing convention
  x_bin_centers = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2
  y_bin_centers = (y_bin_edges[:-1] + y_bin_edges[1:]) / 2
  min_min = min(x_bin_edges[0], y_bin_edges[0])
  max_max = max(x_bin_edges[-1], y_bin_edges[-1])
  if log_prob:
    nonzero_inds = empirical_density != 0
    empirical_density[nonzero_inds] = np.log(empirical_density[nonzero_inds])
    empirical_density[nonzero_inds] -= np.min(empirical_density[nonzero_inds])
    empirical_density[nonzero_inds] /= np.max(empirical_density[nonzero_inds])
    max_emp_prob = np.max(empirical_density)
    gaussian_levels = gen_gaussian_levels(
        0.25*max_emp_prob, 0.9*max_emp_prob, 5, log_probability=True)
  else:
    max_emp_prob = np.max(empirical_density)
    gaussian_levels = gen_gaussian_levels(
        0.25*max_emp_prob, 0.9*max_emp_prob, 5, log_probability=False)
  ax = fig.add_subplot(111, xlim=[min_min, max_max], ylim=[min_min, max_max])
  im = NonUniformImage(ax, interpolation='nearest', cmap='Blues')
  im.set_data(x_bin_centers, y_bin_centers, empirical_density)
  ax.images.append(im)
  if with_contours:
    y_coords, x_coords = np.meshgrid(y_bin_centers,
                                     x_bin_centers, indexing='ij')
    ax.contour(x_coords, y_coords, empirical_density,
               levels=gaussian_levels, colors='k')
  ax.set_title('Joint (log) density', fontsize=10)
  ax.set_ylabel('Values for coefficient 1', fontsize=10)
  ax.set_xlabel('Values for coefficient 0', fontsize=10)
  return fig
