"""
Some simple utilities for plotting our transform codes
"""

import bisect
import numpy as np
from scipy.stats import kurtosis
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D

tab10colors = plt.get_cmap('tab10').colors


def compute_pSNR(target, reconstruction):
  signal_magnitude = np.max(target) - np.min(target)
  MSE = np.mean(np.square(target - reconstruction))
  return 10. * np.log10((signal_magnitude**2)/MSE)


class TrainingLivePlot(object):
  """
  A container for matplotlib plots we'll use to visualize training progress

  Parameters
  ----------
  dict_plot_params : dictionary
      Parameters of the dictionary plot. Currently
      'total_num' : int, 'img_height' : int, 'img_width' : int,
      'plot_width' : int, 'plot_height' : int, 'renorm imgs' : bool,
      'display_ordered' : bool
  code_plot_params : dictionary, optional
      Parameters of the code plot. Currently just 'size' : int
  """
  def __init__(self, dict_plot_params, code_plot_params=None,
               other_plot_params=None):

    plt.ion()

    #################
    # Dictionary plot
    #################
    self.dict_plot_height = dict_plot_params['plot_height']
    self.dict_plot_width = dict_plot_params['plot_width']
    self.img_height = dict_plot_params['img_height']
    self.img_width = dict_plot_params['img_width']
    if dict_plot_params['display_ordered']:
      self.dict_inds = np.arange(self.dict_plot_height*self.dict_plot_width)
    else:
      self.dict_inds = np.random.choice(
          np.arange(dict_plot_params['total_num']),
          self.dict_plot_height*self.dict_plot_width, replace=False)
    self.dict_renorm_flag = dict_plot_params['renorm imgs']
    # prepare a single image to display
    self.dict_h_margin = 2
    self.dict_w_margin = 2
    full_img_height = (self.img_height * self.dict_plot_height +
                       (self.dict_plot_height - 1) * self.dict_h_margin)
    full_img_width = (self.img_width * self.dict_plot_width +
                       (self.dict_plot_width - 1) * self.dict_w_margin)
    composite_img = np.ones((full_img_height, full_img_width))
    for plot_idx in range(len(self.dict_inds)):
      row_idx = plot_idx // self.dict_plot_height
      col_idx = plot_idx % self.dict_plot_width
      pxr1 = row_idx * (self.img_height + self.dict_h_margin)
      pxr2 = pxr1 + self.img_height
      pxc1 = col_idx * (self.img_width + self.dict_w_margin)
      pxc2 = pxc1 + self.img_width
      composite_img[pxr1:pxr2, pxc1:pxc2] = np.zeros(
          (self.img_height, self.img_width))

    self.dict_fig, self.dict_ax = plt.subplots(1, 1, figsize=(10, 10))
    self.dict_fig.suptitle('Random sample of current dictionary', fontsize=15)
    self.temp_imshow_ref = self.dict_ax.imshow(composite_img, cmap='Greys_r')
    self.dict_ax.axis('off')

    self.requires = ['dictionary']

    ###########
    # Code plot
    ###########
    if code_plot_params is not None:
      # set up the code plot
      self.code_size = code_plot_params['size']
      self.num_disp = code_plot_params['num_displayed']
      self.code_fig, self.code_ax = plt.subplots(self.num_disp, 1,
          figsize=(10, self.num_disp/1.5))
      self.code_fig.suptitle('Random sample of codes', fontsize=12)
      self.lineplot_refs = []
      for c_idx in range(self.num_disp):
        _, linerefs, _ = self.code_ax[c_idx].stem(np.arange(self.code_size),
            np.zeros(self.code_size), linefmt='-', markerfmt=' ',
            use_line_collection=False)
        plt.setp(linerefs, 'color', tab10colors[0])
        self.lineplot_refs.append(linerefs)
      self.requires.append('codes')

    ##########################
    # Objective function plots
    ##########################
    self.metrics_fig, self.metrics_ax = plt.subplots(1, 3, figsize=(20, 5))
    # self.sparsity_fig, self.l1_ax = plt.subplots(1, 1, figsize=(10, 5))
    self.recon_sq_err_ax = self.metrics_ax[0]
    self.l1_ax = self.metrics_ax[1]
    self.bpdn_loss_ax = self.metrics_ax[2]
    self.recon_psnr_ax = self.recon_sq_err_ax.twinx()
    self.l0_ax = self.l1_ax.twinx()
    self.recon_sq_err_ref = Line2D([], [], color='k', linewidth=3)
    self.recon_sq_err_saved = []
    self.recon_psnr_ref = Line2D([], [], color='m', linewidth=3)
    self.recon_psnr_saved = []
    self.l1_ref = Line2D([], [], color='b', linewidth=3)
    self.l1_saved = []
    self.l0_ref = Line2D([], [], color='g', linewidth=3)
    self.l0_saved = []
    self.bpdn_loss_ref = Line2D([], [], color='k', linewidth=3)
    self.bpdn_loss_saved = []

    self.recon_sq_err_ax.add_line(self.recon_sq_err_ref)
    self.recon_psnr_ax.add_line(self.recon_psnr_ref)
    self.l1_ax.add_line(self.l1_ref)
    self.l0_ax.add_line(self.l0_ref)
    self.bpdn_loss_ax.add_line(self.bpdn_loss_ref)
    self.sparsity_weight = (1.0 if other_plot_params is None
                            else other_plot_params['sparsity_weight'])

    self.recon_sq_err_min = 0.0
    self.recon_sq_err_max = 100.0
    self.recon_sq_err_ax.set_ylim(self.recon_sq_err_min, self.recon_sq_err_max)
    self.recon_sq_err_ax.set_xlim(0, 100)
    self.recon_sq_err_ax.set_yscale('log')
    self.recon_sq_err_ax.set_ylabel('log(Squared error of reconstructions)', fontsize=10)
    self.recon_sq_err_ax.yaxis.label.set_color('k')
    self.recon_sq_err_ax.set_xlabel('Liveplot visualization iter', fontsize=10)

    self.recon_psnr_min = -10.0
    self.recon_psnr_max = 50.0
    self.recon_psnr_ax.set_ylim(self.recon_psnr_min, self.recon_psnr_max)
    self.recon_psnr_ax.set_xlim(0, 100)
    self.recon_psnr_ax.set_ylabel('pSNR of reconstructions (dB)', fontsize=10)
    self.recon_psnr_ax.yaxis.label.set_color('m')

    self.l1_min = 0.0
    self.l1_max = 100
    self.l1_ax.set_ylim(self.l1_min, self.l1_max)
    self.l1_ax.set_xlim(0, 100)
    self.l1_ax.set_ylabel('log(L1)', fontsize=10)
    self.l1_ax.set_xlabel('Liveplot visualization iter', fontsize=10)
    self.l1_ax.set_yscale('log')
    self.l1_ax.yaxis.label.set_color('b')
    self.l0_min = 0.0
    self.l0_max = 100
    self.l0_ax.set_ylim(self.l0_min, self.l0_max)
    self.l0_ax.set_ylabel('L0', fontsize=10)
    self.l0_ax.yaxis.label.set_color('g')

    self.bpdn_min = 0.0
    self.bpdn_max = 1000
    self.bpdn_loss_ax.set_ylim(self.bpdn_min, self.bpdn_max)
    self.bpdn_loss_ax.set_xlim(0, 100)
    # self.bpdn_loss_ax.set_yscale('log')
    self.bpdn_loss_ax.set_ylabel('log(BPDN loss)', fontsize=10)
    self.bpdn_loss_ax.set_xlabel('Liveplot visualization iter', fontsize=10)
    self.bpdn_loss_ax.yaxis.label.set_color('k')
    self.requires.append('dictionary_codes_and_patches')

  def Requires(self):
    return self.requires

  def ClosePlot(self):
    plt.close()

  def UpdatePlot(self, data, which_plot):

    if which_plot == 'dictionary':
      full_img_height = (self.img_height * self.dict_plot_height +
                         (self.dict_plot_height - 1) * self.dict_h_margin)
      full_img_width = (self.img_width * self.dict_plot_width +
                         (self.dict_plot_width - 1) * self.dict_w_margin)
      if self.dict_renorm_flag:
        maximum_value = 1.0
      else:
        maximum_value = np.max(data[:, self.dict_inds])

      composite_img = maximum_value * np.ones((full_img_height, full_img_width))
      for plot_idx in range(len(self.dict_inds)):
        if self.dict_renorm_flag:
          this_filter = data[:, self.dict_inds[plot_idx]]
          this_filter = this_filter - np.min(this_filter)
          this_filter = this_filter / np.max(this_filter)  # now in [0, 1]
        else:
          this_filter = np.copy(data[:, self.dict_inds[plot_idx]])

        row_idx = plot_idx // self.dict_plot_height
        col_idx = plot_idx % self.dict_plot_width
        pxr1 = row_idx * (self.img_height + self.dict_h_margin)
        pxr2 = pxr1 + self.img_height
        pxc1 = col_idx * (self.img_width + self.dict_w_margin)
        pxc2 = pxc1 + self.img_width
        composite_img[pxr1:pxr2, pxc1:pxc2] = np.reshape(
            this_filter, (self.img_height, self.img_width))

      self.dict_ax.clear()
      self.dict_ax.imshow(composite_img, cmap='Greys_r')
      self.dict_ax.axis('off')
      plt.pause(0.1)

    elif which_plot == 'codes':
      assert data.shape[1] >= self.num_disp
      rand_inds = np.random.choice(np.arange(data.shape[1]), self.num_disp,
                                   replace=False)
      for d_idx in range(self.num_disp):
        min_code_val = np.min(data[:, rand_inds[d_idx]])
        max_code_val = np.max(data[:, rand_inds[d_idx]])
        for l_idx in range(self.code_size):
          self.lineplot_refs[d_idx][l_idx].set_ydata(
              [0.0, data[l_idx, rand_inds[d_idx]]])
        self.code_ax[d_idx].set_ylim(min_code_val, max_code_val)
      plt.pause(0.1)

    elif which_plot == 'dictionary_codes_and_patches':
      temp = np.dot(data['dictionary'], data['codes'])
      self.recon_sq_err_saved.append(0.5*np.mean(np.square(data['patches'] - temp)))
      self.recon_psnr_saved.append(compute_pSNR(data['patches'], temp))
      self.l1_saved.append(np.mean(np.sum(np.abs(data['codes']), axis=-1)))
      self.l0_saved.append(np.mean(np.sum(data['codes'] != 0.0, axis=0)))
      self.bpdn_loss_saved.append(self.recon_sq_err_saved[-1] +
                                  self.sparsity_weight * self.l1_saved[-1])
      self.recon_sq_err_ref.set_data(np.arange(len(self.recon_sq_err_saved)),
                                     self.recon_sq_err_saved)
      self.recon_psnr_ref.set_data(np.arange(len(self.recon_psnr_saved)),
                                   self.recon_psnr_saved)
      self.l1_ref.set_data(np.arange(len(self.l1_saved)), self.l1_saved)
      self.l0_ref.set_data(np.arange(len(self.l0_saved)), self.l0_saved)
      self.bpdn_loss_ref.set_data(np.arange(len(self.bpdn_loss_saved)),
                                  self.bpdn_loss_saved)
      self.recon_sq_err_min = min(self.recon_sq_err_saved)
      self.recon_sq_err_max = max(self.recon_sq_err_saved)
      self.recon_psnr_min = min(self.recon_psnr_saved)
      self.recon_psnr_max = max(self.recon_psnr_saved)
      self.l1_min = min(self.l1_saved)
      self.l1_max = max(self.l1_saved)
      self.l0_min = min(self.l0_saved)
      self.l0_max = max(self.l0_saved)
      self.bpdn_min = min(self.bpdn_loss_saved)
      self.bpdn_max = max(self.bpdn_loss_saved)
      self.recon_sq_err_ax.set_ylim(self.recon_sq_err_min, self.recon_sq_err_max)
      self.recon_psnr_ax.set_ylim(self.recon_psnr_min, self.recon_psnr_max)
      self.l1_ax.set_ylim(self.l1_min, self.l1_max)
      self.l0_ax.set_ylim(self.l0_min, self.l0_max)
      self.bpdn_loss_ax.set_ylim(self.bpdn_min, self.bpdn_max)
      self.recon_sq_err_ax.set_xlim(0, len(self.recon_sq_err_saved))
      self.recon_psnr_ax.set_xlim(0, len(self.recon_psnr_saved))
      self.l1_ax.set_xlim(0, len(self.l1_saved))
      self.l0_ax.set_xlim(0, len(self.l0_saved))
      self.bpdn_loss_ax.set_xlim(0, len(self.bpdn_loss_saved))
      plt.pause(0.1)


def get_dictionary_tile_imgs(dictionary, image_dims, renormalize):
  """
  Arranges a dictionary into a series of imgs that tile elements side by side

  Parameters
  ----------
  dictionary : ndarray(float32, size=(n, s))
      A dictionary, the matrix used in the linear synthesis transform
  image_dims : tuple(int, int)
      The dimension of each patch - used to visualize the dictionary elements
      as an image patch.
  renormalize : bool
      If present, renormalize the pixel values for each basis function to the
      interval [0, 1]. Otherwise they are left on their original scale.

  Returns
  -------
  tile_imgs : list(ndarray)
      Each element is an image to be displayed by imshow
  tile_img_ranges : list(tuple(float, float))
      The range that is supposed to be used by imshow when displaying each
      image. Helps to maintain the right contrast if we didn't renormalize
      each dictionary element.
  """
  max_de_val = np.max(dictionary)
  min_de_val = np.min(dictionary)

  max_de_per_img = 80*80  # max 80x80 {d}ictionary {e}lements per tile img
  assert np.sqrt(max_de_per_img) % 1 == 0, 'please pick a square number'
  num_de = dictionary.shape[1]
  num_tile_imgs = int(np.ceil(num_de / max_de_per_img))
  # this determines how many dictionary elements are aranged in a square
  # grid within any given img
  if num_tile_imgs > 1:
    de_per_img = max_de_per_img
  else:
    squares = [x**2 for x in range(1, int(np.sqrt(max_de_per_img))+1)]
    de_per_img = squares[bisect.bisect_left(squares, num_de)]
  plot_sidelength = int(np.sqrt(de_per_img))

  h_margin = 2
  w_margin = 2
  full_img_height = (image_dims[0] * plot_sidelength +
                     (plot_sidelength - 1) * h_margin)
  full_img_width = (image_dims[1] * plot_sidelength +
                    (plot_sidelength - 1) * w_margin)

  de_idx = 0
  tile_imgs = []
  tile_img_ranges = []
  for in_de_img_idx in range(num_tile_imgs):
    if renormalize:
      composite_img = np.ones((full_img_height, full_img_width))
    else:
      composite_img = max_de_val * np.ones((full_img_height, full_img_width))

    img_de_idx = de_idx % de_per_img
    while img_de_idx < de_per_img and de_idx < num_de:

      if renormalize:
        this_de = dictionary[:, de_idx]
        this_de = this_de - np.min(this_de)
        this_de = this_de / np.max(this_de)  # now in [0, 1]
      else:
        this_de = np.copy(dictionary[:, de_idx])

      # okay, now actually plot the DEs in this tile image
      row_idx = img_de_idx // plot_sidelength
      col_idx = img_de_idx % plot_sidelength
      pxr1 = row_idx * (image_dims[0] + h_margin)
      pxr2 = pxr1 + image_dims[0]
      pxc1 = col_idx * (image_dims[1] + w_margin)
      pxc2 = pxc1 + image_dims[1]
      composite_img[pxr1:pxr2, pxc1:pxc2] = this_de.reshape((image_dims[0],
                                                             image_dims[1]))

      img_de_idx += 1
      de_idx += 1

    min_plot_val = 0. if renormalize else min_de_val
    max_plot_val = 1. if renormalize else max_de_val

    tile_imgs.append(composite_img)
    tile_img_ranges.append((min_plot_val, max_plot_val))

  return tile_imgs, tile_img_ranges


def display_dictionary(dictionary, image_dims, plot_title="", renormalize=True):
  """
  Plot each of the dictionary elements side by side

  Parameters
  ----------
  dictionary : ndarray(float32, size=(n, s))
      A dictionary, the matrix used in the linear synthesis transform
  image_dims : tuple(int, int)
      The dimension of each patch - used to visualize the dictionary elements
      as an image patch.
  plot_title : str, optional
      The title of the plot. Default ""
  renormalize : bool, optional
      If present, renormalize each basis function to the interval [0, 1] before
      displaying. Otherwise they are displayed on their original scale. Default
      True

  Returns
  -------
  dictionary_figs : list
      A list containing pyplot figures. Can be saved separately, or whatever
      from the calling function
  """
  t_ims, im_ranges = get_dictionary_tile_imgs(
      dictionary, image_dims, renormalize=renormalize)
  fig_refs = []
  for fig_idx in range(len(t_ims)):
    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(plot_title + ', fig {} of {}'.format(
                 fig_idx+1, len(t_ims)), fontsize=15)
    plt.imshow(t_ims[fig_idx], cmap='Greys_r', vmin=im_ranges[fig_idx][0],
               vmax=im_ranges[fig_idx][1], interpolation='nearest')
    plt.axis('off')

    fig_refs.append(fig)

  return fig_refs


def display_codes(codes, plot_title=""):
  """
  Plot each of codes separately as a stem plot

  Parameters
  ----------
  codes : ndarray(float32, size=(s, B))
      The codes for a batch of size B. B shouldn't be too large unless you want
      to wait around all day for this to finish.
  plot_title : str, optional
      The title of the plot. Default ""

  Returns
  -------
  code_figs : list
      A list containing pyplot figures. Can be saved separately, or whatever
      from the calling function
  """
  code_size = codes.shape[0]
  num_codes = codes.shape[1]
  max_data_pt_per_fig = 40
  num_code_figs = int(np.ceil(num_codes / max_data_pt_per_fig))
  if num_code_figs > 1:
    data_pt_per_fig = max_data_pt_per_fig
  else:
    data_pt_per_fig = num_codes

  dpt_idx = 0
  code_figs = []
  for in_code_fig_idx in range(num_code_figs):
    fig, ax = plt.subplots(data_pt_per_fig, 1, figsize=(10, num_codes/1.5))
    fig.suptitle(plot_title + ', fig {} of {}'.format(
                 in_code_fig_idx+1, num_code_figs), fontsize=15)
    fig_dpt_idx = dpt_idx % data_pt_per_fig
    while fig_dpt_idx < data_pt_per_fig and dpt_idx < num_codes:
      if dpt_idx % 25 == 0:
        print('plotted', dpt_idx, 'of', num_codes, 'codes')
      _, linerefs, _ = ax[fig_dpt_idx].stem(np.arange(code_size),
          codes[:, dpt_idx], linefmt='-', markerfmt=' ')
      plt.setp(linerefs, 'color', tab10colors[0])
      fig_dpt_idx += 1
      dpt_idx += 1
    code_figs.append(fig)

  return code_figs


def display_code_marginal_densities(codes, num_hist_bins,
    lines=True, overlaid=False, plot_title=""):
  """
  Estimate the marginal density of the coefficients of a code over some dataset

  Parameters
  ----------
  codes : ndarray(float32, size=(s, D))
      The codes for a dataset of size D. These are the vectors x for each sample
      from the dataset. The value s is the dimensionality of the code
  num_hist_bins : int
      The number of bins to use when we make a histogram estimate of the
      empirical density.
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
  if overlaid:
    # there's just a single plot
    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(plot_title, fontsize=15)
    ax = plt.subplot(1, 1, 1)
    blue=plt.get_cmap('Blues')
    cmap_indeces = np.linspace(0.25, 1.0, codes.shape[0])

    histogram_min = np.min(codes)
    histogram_max = np.max(codes)
    histogram_bin_edges = np.linspace(histogram_min, histogram_max,
                                      num_hist_bins + 1)
    histogram_bin_centers = (histogram_bin_edges[:-1] +
                             histogram_bin_edges[1:]) / 2
    for de_idx in range(codes.shape[0]):
      counts, _ = np.histogram(codes[de_idx], histogram_bin_edges)
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
    de_figs = [fig]

  else:
    # every coefficient gets its own subplot
    max_de_per_fig = 80*80  # max 80x80 {d}ictionary {e}lements displayed each fig
    assert np.sqrt(max_de_per_fig) % 1 == 0, 'please pick a square number'
    num_de = codes.shape[0]
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
        histogram_min = min(codes[de_idx, :])
        histogram_max = max(codes[de_idx, :])
        histogram_bin_edges = np.linspace(histogram_min, histogram_max,
                                          num_hist_bins + 1)
        histogram_bin_centers = (histogram_bin_edges[:-1] +
                                 histogram_bin_edges[1:]) / 2
        counts, _ = np.histogram(codes[de_idx, :], histogram_bin_edges)
        empirical_density = counts / np.sum(counts)
        max_density = np.max(empirical_density)
        variance = np.var(codes[de_idx, :])
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
        ax.set_xticks([histogram_min, 0., histogram_max])
        ax.set_yticks([0., max_density])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.set_yscale('log')

        ax.text(0.1, 0.97, '{:.2f}'.format(variance), horizontalalignment='left',
                verticalalignment='top', transform=ax.transAxes, color='b',
                fontsize=5)
        ax.text(0.1, 0.8, '{:.2f}'.format(hist_kurtosis),
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, color='r', fontsize=5)

        fig.add_subplot(ax)
        fig_de_idx += 1
        de_idx += 1
      de_figs.append(fig)

  return de_figs
