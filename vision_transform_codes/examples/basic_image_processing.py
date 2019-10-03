"""
Just some simple examples of how to use the image processing utils
"""

import sys
import os
examples_fullpath = os.path.dirname(os.path.abspath(__file__))
toplevel_dir_fullpath = examples_fullpath[:examples_fullpath.rfind('/')+1]
sys.path.insert(0, toplevel_dir_fullpath)

import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import utils.image_processing as im_proc
import utils.plotting as plot_utils

def main():
  # we'll just show this on Kodak images, you can change this as you see fit
  WHICH_DEMO_IMAGES = 'Kodak'

  if WHICH_DEMO_IMAGES == 'Kodak':
    raw_data_filepath = \
        '/home/spencerkent/Datasets/Kodak/kodak_full_images_training.p'
    unprocessed_images = pickle.load(open(raw_data_filepath, 'rb'))
    unprocessed_images = [x.astype('float32') for x in unprocessed_images]
  else:
    raise KeyError('not implemented, ' +
                   'see create_patch_training_set in utils/image_processing.py')

  ##########################
  # Low-pass filter an image
  ##########################
  orig_img = unprocessed_images[4]  # arbitrary
  dft_num_samples = orig_img.shape
  lpf = im_proc.get_low_pass_filter(
      dft_num_samples, {'shape': 'exponential', 'cutoff': 0.1, 'order': 4.0})
  lpf_img = im_proc.filter_image(orig_img, lpf)
  orig_img_recovered = im_proc.filter_image(lpf_img, 1./lpf)
  # A little visualization
  visualize_lp_filtering(orig_img, lpf_img, lpf,
                         orig_img_recovered, dft_num_samples)

  ################################################
  # Whiten with 'Atick and Redlich' whitening, the
  # whitening originally used in sparse coding
  ################################################
  orig_img = unprocessed_images[4]  # arbitrary
  dft_num_samples = orig_img.shape
  white_img, white_filt = im_proc.whiten_center_surround(
      orig_img, return_filter=True)
  orig_img_recovered = im_proc.unwhiten_center_surround(white_img, white_filt)
  # A little visualization
  visualize_AR_whitening(orig_img, white_img, white_filt,
                         orig_img_recovered, dft_num_samples)

  #############################
  # Whiten with 'ZCA' whitening
  #############################
  # we have to estimate the ZCA_transform on a large batch of data, and we do
  # it for 8x8 patches. Rather than whitening whole images, we just whiten
  # patches and then we can go back and reassemble the image.
  print('Creating a dataset of patches')
  one_mil_image_patches = im_proc.create_patch_training_set(
      ['patch'], (8, 8),
      1000000, 1, edge_buffer=5, dataset='Kodak',
      datasetparams={'filepath': raw_data_filepath,
                     'exclude': []})['batched_patches'][0]
  print('Estimating the ZCA transform parameters based on this data')
  _, ZCA_params = im_proc.whiten_ZCA(one_mil_image_patches)

  print('Applying transform to test image')
  orig_img = unprocessed_images[4]  # arbitrary
  orig_img_patches, orig_img_patch_pos = im_proc.patches_from_single_image(
      orig_img, (8, 8))
  white_patches = im_proc.whiten_ZCA(orig_img_patches, ZCA_params)
  white_img = im_proc.assemble_image_from_patches(
      white_patches, (8, 8), orig_img_patch_pos)
  orig_img_recovered_patches = im_proc.unwhiten_ZCA(white_patches, ZCA_params)
  orig_img_recovered = im_proc.assemble_image_from_patches(
      orig_img_recovered_patches, (8, 8), orig_img_patch_pos)
  # A little visualization
  visualize_ZCA_whitening(orig_img, white_img, ZCA_params, orig_img_recovered)



  plt.show()



def visualize_lp_filtering(o_img, lp_img, lpf_filt,
                           o_img_recovered, dft_nsamps):
  fig = plt.figure(figsize=(15, 8), dpi=100)
  fig.suptitle('Low-pass filtering', fontsize=12)
  sp_colwidths = [1, 1, 1, 1]
  sp_rowheights = [4, 2, 2]
  gridspec = fig.add_gridspec(ncols=4, nrows=3, width_ratios=sp_colwidths,
                              height_ratios=sp_rowheights)

  ax = fig.add_subplot(gridspec[0, 0])
  ax.set_title('Original image', fontsize=10)
  ax.imshow(o_img, cmap='Greys_r', vmin=o_img.min(), vmax=o_img.max())

  ax = fig.add_subplot(gridspec[0, 1])
  ax.set_title('Low-pass filtered image', fontsize=10)
  ax.imshow(lp_img, cmap='Greys_r', vmin=o_img.min(), vmax=o_img.max())
  ax.set_xlabel('pSNR to orig: ' + '{:.2f}'.format(
    plot_utils.compute_pSNR(o_img, lp_img)) + 'dB')

  ax = fig.add_subplot(gridspec[0, 2])
  ax.set_title('Recovered Inverted lpf image', fontsize=10)
  ax.imshow(o_img_recovered, cmap='Greys_r',
            vmin=o_img.min(), vmax=o_img.max())
  ax.set_xlabel('pSNR to orig: ' + '{:.2f}'.format(
    plot_utils.compute_pSNR(o_img, o_img_recovered)) + 'dB')

  ax = fig.add_subplot(gridspec[0, 3])
  ax.set_title('Difference image', fontsize=10)
  ax.imshow(o_img - o_img_recovered, cmap='Greys_r',
             vmin=o_img.min(), vmax=o_img.max())

  ax = fig.add_subplot(gridspec[1, 0])
  ax.set_title('(log) magnitude of 2D DFT\noriginal image', fontsize=10)
  orig_dft_im = ax.imshow(
      np.fft.fftshift(np.log(np.abs(np.fft.fft2(o_img, dft_nsamps)))),
      cmap='magma', aspect=dft_nsamps[1]/dft_nsamps[0])
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=-0.2)
  plt.colorbar(orig_dft_im, cax=cax)
  ax.set_yticks([0, dft_nsamps[0]//2, dft_nsamps[0]])
  ax.set_xticks([0, dft_nsamps[1]//2, dft_nsamps[1]])
  ax.set_yticklabels(['-0.5', '0.0', '0.5'])
  ax.set_xticklabels(['-0.5', '0.0', '0.5'])

  ax = fig.add_subplot(gridspec[1, 1])
  ax.set_title('(log) magnitude of 2D DFT\nlp-filtered image', fontsize=10)
  lp_dft_im = ax.imshow(
      np.fft.fftshift(np.log(np.abs(np.fft.fft2(lp_img, dft_nsamps)))),
      cmap='magma', aspect=dft_nsamps[1]/dft_nsamps[0])
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=-0.2)
  plt.colorbar(lp_dft_im, cax=cax)
  ax.set_yticks([0, dft_nsamps[0]//2, dft_nsamps[0]])
  ax.set_xticks([0, dft_nsamps[1]//2, dft_nsamps[1]])
  ax.set_yticklabels(['-0.5', '0.0', '0.5'])
  ax.set_xticklabels(['-0.5', '0.0', '0.5'])

  ax = fig.add_subplot(gridspec[2, 0])
  ax.set_title('magnitude of 2D DFT\nlow-pass filter', fontsize=10)
  imax = ax.imshow(np.fft.fftshift(np.abs(lpf_filt)), cmap='magma',
                    aspect=dft_nsamps[1]/dft_nsamps[0])
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=-0.2)
  plt.colorbar(imax, cax=cax)
  ax.set_yticks([0, dft_nsamps[0]//2, dft_nsamps[0]])
  ax.set_xticks([0, dft_nsamps[1]//2, dft_nsamps[1]])
  ax.set_yticklabels(['-0.5', '0.0', '0.5'])
  ax.set_xticklabels(['-0.5', '0.0', '0.5'])

  ax = fig.add_subplot(gridspec[2, 1])
  ax.set_title('Slice along vert axis', fontsize=10)
  ax.plot(np.linspace(-0.5, 0.5, lpf_filt.shape[0]),
          np.fft.fftshift(np.abs(lpf_filt[:, 0])))
  ax.set_xlabel('Frequency (units of sampling freq)', fontsize=8)
  ax.set_ylabel('Magnitude of filter', fontsize=8)

  ax = fig.add_subplot(gridspec[2, 2])
  ax.set_title('Slice along horz axis', fontsize=10)
  ax.plot(np.linspace(-0.5, 0.5, lpf_filt.shape[1]),
          np.fft.fftshift(np.abs(lpf_filt[0, :])))
  ax.set_xlabel('Frequency (units of sampling freq)', fontsize=8)
  ax.set_ylabel('Magnitude of filter', fontsize=8)

  ax = fig.add_subplot(gridspec[2, 3])
  ax.set_title('Filter in image space', fontsize=10)
  uncropped = np.fft.fftshift(np.real(np.fft.ifft2(lpf_filt)))
  vmiddle = uncropped.shape[0]//2
  hmiddle = uncropped.shape[1]//2
  cropped = uncropped[vmiddle-20:vmiddle+21, hmiddle-20:hmiddle+21]
  imax = ax.imshow(cropped, cmap='Greys_r')
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.1)
  plt.colorbar(imax, cax=cax)
  ax.set_yticks([0, 20, 40])
  ax.set_xticks([0, 20, 40])
  ax.set_yticklabels(['-20', '0', '20'])
  ax.set_xticklabels(['-20', '0', '20'])
  plt.tight_layout()


def visualize_AR_whitening(o_img, w_img, w_filt, o_img_recovered, dft_nsamps):
  fig = plt.figure(figsize=(15, 8), dpi=100)
  fig.suptitle('\'Atick and Redlich\' whitening', fontsize=12)
  sp_colwidths = [1, 1, 1, 1]
  sp_rowheights = [4, 2, 2]
  gridspec = fig.add_gridspec(ncols=4, nrows=3, width_ratios=sp_colwidths,
                              height_ratios=sp_rowheights)

  ax = fig.add_subplot(gridspec[0, 0])
  ax.set_title('Original image', fontsize=10)
  ax.imshow(o_img, cmap='Greys_r', vmin=o_img.min(), vmax=o_img.max())

  ax = fig.add_subplot(gridspec[0, 1])
  ax.set_title('Whitened image', fontsize=10)
  ax.imshow(w_img, cmap='Greys_r')
  ax.set_xlabel('pSNR to orig: ' + '{:.2f}'.format(
    plot_utils.compute_pSNR(o_img, w_img)) + 'dB')

  ax = fig.add_subplot(gridspec[0, 2])
  ax.set_title('Recovered inverse whitened image', fontsize=10)
  ax.imshow(o_img_recovered, cmap='Greys_r',
            vmin=o_img.min(), vmax=o_img.max())
  ax.set_xlabel('pSNR to orig: ' + '{:.2f}'.format(
    plot_utils.compute_pSNR(o_img, o_img_recovered)) + 'dB')

  ax = fig.add_subplot(gridspec[0, 3])
  ax.set_title('Difference image', fontsize=10)
  ax.imshow(o_img - o_img_recovered, cmap='Greys_r',
            vmin=o_img.min(), vmax=o_img.max())

  ax = fig.add_subplot(gridspec[1, 0])
  ax.set_title('(log) magnitude of 2D DFT\noriginal image', fontsize=10)
  orig_dft_im = plt.imshow(
      np.fft.fftshift(np.log(np.abs(np.fft.fft2(o_img, dft_nsamps)))),
      cmap='magma', aspect=dft_nsamps[1]/dft_nsamps[0])
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=-0.2)
  plt.colorbar(orig_dft_im, cax=cax)
  ax.set_yticks([0, dft_nsamps[0]//2, dft_nsamps[0]])
  ax.set_xticks([0, dft_nsamps[1]//2, dft_nsamps[1]])
  ax.set_yticklabels(['-0.5', '0.0', '0.5'])
  ax.set_xticklabels(['-0.5', '0.0', '0.5'])

  ax = fig.add_subplot(gridspec[1, 1])
  ax.set_title('(log) magnitude of 2D DFT\nwhitened image', fontsize=10)
  orig_dft_im = plt.imshow(
      np.fft.fftshift(np.log(np.abs(np.fft.fft2(w_img, dft_nsamps)))),
      cmap='magma', aspect=dft_nsamps[1]/dft_nsamps[0])
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=-0.2)
  plt.colorbar(orig_dft_im, cax=cax)
  ax.set_yticks([0, dft_nsamps[0]//2, dft_nsamps[0]])
  ax.set_xticks([0, dft_nsamps[1]//2, dft_nsamps[1]])
  ax.set_yticklabels(['-0.5', '0.0', '0.5'])
  ax.set_xticklabels(['-0.5', '0.0', '0.5'])

  ax = fig.add_subplot(gridspec[1, 2])
  ax.set_title('Joint density of adjacent pixels\noriginal image', fontsize=10)
  # estimate the joint distribution of any arbitrary pair of adjacent pixels
  num_rand_samps = 10000
  samps = []
  for _ in range(num_rand_samps):
    pix1 = [np.random.randint(low=1, high=o_img.shape[0]-1),
            np.random.randint(low=1, high=o_img.shape[1]-1)]
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1), (0, 1),
               (1, -1), (1, 0), (1, 1)]
    offset = offsets[np.random.choice(np.arange(8))]
    pix2 = (pix1[0] + offset[0], pix1[1] + offset[1])
    samps.append([o_img[pix1[0], pix1[1]], o_img[pix2[0], pix2[1]]])
  samps = np.array(samps)
  hist_nbins = (30, 30)
  counts, hist_bin_edges = np.histogramdd(samps, bins=hist_nbins)
  empirical_density = counts / np.sum(counts)
  log_density = np.copy(empirical_density)
  nonzero_inds = log_density != 0
  log_density[nonzero_inds] = np.log(log_density[nonzero_inds])
  log_density[nonzero_inds] -= np.min(log_density[nonzero_inds])
  log_density[nonzero_inds] /= np.max(log_density[nonzero_inds])
  hist_bin_centers = [(hist_bin_edges[x][:-1] + hist_bin_edges[x][1:]) / 2
                      for x in range(len(hist_bin_edges))]
  min_max_x = [hist_bin_centers[0][0], hist_bin_centers[0][-1]]
  min_max_y = [hist_bin_centers[1][0], hist_bin_centers[1][-1]]
  plt.imshow(np.flip(log_density.T, axis=0),
             interpolation='nearest', cmap='Blues')
  plt.xlabel('Values for a random pixel', fontsize=8)
  plt.ylabel('Values for random adjacent pixel', fontsize=8)
  plt.gca().set_xticks([0.0, hist_nbins[0]-1])
  plt.gca().set_yticks([hist_nbins[1]-1, 0.0])
  plt.xlim([-0.5, hist_nbins[0]-0.5])
  plt.ylim([hist_nbins[1]-0.5, -0.5])
  plt.gca().set_xticklabels(['{:.1f}'.format(hist_bin_centers[0][0]),
                             '{:.1f}'.format(hist_bin_centers[0][-1])],
                            fontsize=7)
  plt.gca().set_yticklabels(['{:.1f}'.format(hist_bin_centers[1][0]),
                             '{:.1f}'.format(hist_bin_centers[1][-1])],
                            fontsize=7)
  plt.gca().set_aspect('equal')

  ax = fig.add_subplot(gridspec[1, 3])
  ax.set_title('Joint density of adjacent pixels\nwhitened image', fontsize=10)
  # estimate the joint distribution of any arbitrary pair of adjacent pixels
  num_rand_samps = 10000
  samps = []
  for _ in range(num_rand_samps):
    pix1 = [np.random.randint(low=1, high=w_img.shape[0]-1),
            np.random.randint(low=1, high=w_img.shape[1]-1)]
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1), (0, 1),
               (1, -1), (1, 0), (1, 1)]
    offset = offsets[np.random.choice(np.arange(8))]
    pix2 = (pix1[0] + offset[0], pix1[1] + offset[1])
    samps.append([w_img[pix1[0], pix1[1]], w_img[pix2[0], pix2[1]]])
  samps = np.array(samps)
  hist_nbins = (30, 30)
  counts, hist_bin_edges = np.histogramdd(samps, bins=hist_nbins)
  empirical_density = counts / np.sum(counts)
  log_density = np.copy(empirical_density)
  nonzero_inds = log_density != 0
  log_density[nonzero_inds] = np.log(log_density[nonzero_inds])
  log_density[nonzero_inds] -= np.min(log_density[nonzero_inds])
  log_density[nonzero_inds] /= np.max(log_density[nonzero_inds])
  hist_bin_centers = [(hist_bin_edges[x][:-1] + hist_bin_edges[x][1:]) / 2
                      for x in range(len(hist_bin_edges))]
  min_max_x = [hist_bin_centers[0][0], hist_bin_centers[0][-1]]
  min_max_y = [hist_bin_centers[1][0], hist_bin_centers[1][-1]]
  plt.imshow(np.flip(log_density.T, axis=0),
             interpolation='nearest', cmap='Blues')
  plt.xlabel('Values for a random pixel', fontsize=8)
  plt.ylabel('Values for random adjacent pixel', fontsize=8)
  plt.gca().set_xticks([0.0, hist_nbins[0]-1])
  plt.gca().set_yticks([hist_nbins[1]-1, 0.0])
  plt.xlim([-0.5, hist_nbins[0]-0.5])
  plt.ylim([hist_nbins[1]-0.5, -0.5])
  plt.gca().set_xticklabels(['{:.1f}'.format(hist_bin_centers[0][0]),
                             '{:.1f}'.format(hist_bin_centers[0][-1])],
                            fontsize=7)
  plt.gca().set_yticklabels(['{:.1f}'.format(hist_bin_centers[1][0]),
                             '{:.1f}'.format(hist_bin_centers[1][-1])],
                            fontsize=7)
  plt.gca().set_aspect('equal')

  ax = fig.add_subplot(gridspec[2, 0])
  ax.set_title('magnitude of 2D DFT\nwhitening filter', fontsize=10)
  imax = plt.imshow(np.fft.fftshift(np.abs(w_filt)), cmap='magma',
                    aspect=dft_nsamps[1]/dft_nsamps[0])
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=-0.2)
  plt.colorbar(imax, cax=cax)
  ax.set_yticks([0, dft_nsamps[0]//2, dft_nsamps[0]])
  ax.set_xticks([0, dft_nsamps[1]//2, dft_nsamps[1]])
  ax.set_yticklabels(['-0.5', '0.0', '0.5'])
  ax.set_xticklabels(['-0.5', '0.0', '0.5'])

  ax = fig.add_subplot(gridspec[2, 1])
  plt.title('Slice along vert axis', fontsize=10)
  plt.plot(np.linspace(-0.5, 0.5, w_filt.shape[0]),
           np.fft.fftshift(np.abs(w_filt[:, 0])))
  plt.xlabel('Frequency (units of sampling freq)', fontsize=8)
  plt.ylabel('Magnitude of filter', fontsize=8)

  ax = fig.add_subplot(gridspec[2, 2])
  plt.title('Slice along horz axis', fontsize=10)
  plt.plot(np.linspace(-0.5, 0.5, w_filt.shape[1]),
           np.fft.fftshift(np.abs(w_filt[0, :])))
  plt.xlabel('Frequency (units of sampling freq)', fontsize=8)
  plt.ylabel('Magnitude of filter', fontsize=8)

  ax = fig.add_subplot(gridspec[2, 3])
  plt.title('Filter in image space', fontsize=10)
  uncropped = np.fft.fftshift(np.real(np.fft.ifft2(w_filt)))
  vmiddle = uncropped.shape[0]//2
  hmiddle = uncropped.shape[1]//2
  cropped = uncropped[vmiddle-20:vmiddle+21, hmiddle-20:hmiddle+21]
  imax = plt.imshow(cropped, cmap='Greys_r')
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.1)
  plt.colorbar(imax, cax=cax)
  ax.set_yticks([0, 20, 40])
  ax.set_xticks([0, 20, 40])
  ax.set_yticklabels(['-20', '0', '20'])
  ax.set_xticklabels(['-20', '0', '20'])

  plt.tight_layout()


def visualize_ZCA_whitening(o_img, w_img, ZCA, o_img_recovered):
  fig = plt.figure(figsize=(15, 8), dpi=100)
  fig.suptitle('ZCA whitening', fontsize=12)
  sp_colwidths = [1, 1, 1, 1]
  sp_rowheights = [4, 2, 2]
  gridspec = fig.add_gridspec(ncols=4, nrows=3, width_ratios=sp_colwidths,
                              height_ratios=sp_rowheights)

  ax = fig.add_subplot(gridspec[0, 0])
  ax.set_title('Original image', fontsize=10)
  ax.imshow(o_img, cmap='Greys_r', vmin=o_img.min(), vmax=o_img.max())

  ax = fig.add_subplot(gridspec[0, 1])
  ax.set_title('Whitened image', fontsize=10)
  ax.imshow(w_img, cmap='Greys_r')
  ax.set_xlabel('pSNR to orig: ' + '{:.2f}'.format(
    plot_utils.compute_pSNR(o_img, w_img)) + 'dB')

  ax = fig.add_subplot(gridspec[0, 2])
  ax.set_title('Recovered inverse whitened image', fontsize=10)
  ax.imshow(o_img_recovered, cmap='Greys_r',
            vmin=o_img.min(), vmax=o_img.max())
  ax.set_xlabel('pSNR to orig: ' + '{:.2f}'.format(
    plot_utils.compute_pSNR(o_img, o_img_recovered)) + 'dB')

  ax = fig.add_subplot(gridspec[0, 3])
  ax.set_title('Difference image', fontsize=10)
  ax.imshow(o_img - o_img_recovered, cmap='Greys_r',
            vmin=o_img.min(), vmax=o_img.max())

  ax = fig.add_subplot(gridspec[1, 0])
  ax.set_title('(log) magnitude of 2D DFT\noriginal image', fontsize=10)
  orig_dft_im = plt.imshow(
      np.fft.fftshift(np.log(np.abs(np.fft.fft2(o_img)))),
      cmap='magma', aspect=o_img.shape[1]/o_img.shape[0])
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=-0.2)
  plt.colorbar(orig_dft_im, cax=cax)
  ax.set_yticks([0, o_img.shape[0]//2, o_img.shape[0]])
  ax.set_xticks([0, o_img.shape[1]//2, o_img.shape[1]])
  ax.set_yticklabels(['-0.5', '0.0', '0.5'])
  ax.set_xticklabels(['-0.5', '0.0', '0.5'])

  ax = fig.add_subplot(gridspec[1, 1])
  ax.set_title('(log) magnitude of 2D DFT\nwhitened image', fontsize=10)
  orig_dft_im = plt.imshow(
      np.fft.fftshift(np.log(np.abs(np.fft.fft2(w_img)))),
      cmap='magma', aspect=w_img.shape[1]/w_img.shape[0])
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=-0.2)
  plt.colorbar(orig_dft_im, cax=cax)
  ax.set_yticks([0, w_img.shape[0]//2, w_img.shape[0]])
  ax.set_xticks([0, w_img.shape[1]//2, w_img.shape[1]])
  ax.set_yticklabels(['-0.5', '0.0', '0.5'])
  ax.set_xticklabels(['-0.5', '0.0', '0.5'])

  ax = fig.add_subplot(gridspec[1, 2])
  ax.set_title('Joint density of adjacent pixels\noriginal image', fontsize=10)
  # estimate the joint distribution of any arbitrary pair of adjacent pixels
  num_rand_samps = 10000
  samps = []
  for _ in range(num_rand_samps):
    pix1 = [np.random.randint(low=1, high=o_img.shape[0]-1),
            np.random.randint(low=1, high=o_img.shape[1]-1)]
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1), (0, 1),
               (1, -1), (1, 0), (1, 1)]
    offset = offsets[np.random.choice(np.arange(8))]
    pix2 = (pix1[0] + offset[0], pix1[1] + offset[1])
    samps.append([o_img[pix1[0], pix1[1]], o_img[pix2[0], pix2[1]]])
  samps = np.array(samps)
  hist_nbins = (30, 30)
  counts, hist_bin_edges = np.histogramdd(samps, bins=hist_nbins)
  empirical_density = counts / np.sum(counts)
  log_density = np.copy(empirical_density)
  nonzero_inds = log_density != 0
  log_density[nonzero_inds] = np.log(log_density[nonzero_inds])
  log_density[nonzero_inds] -= np.min(log_density[nonzero_inds])
  log_density[nonzero_inds] /= np.max(log_density[nonzero_inds])
  hist_bin_centers = [(hist_bin_edges[x][:-1] + hist_bin_edges[x][1:]) / 2
                      for x in range(len(hist_bin_edges))]
  min_max_x = [hist_bin_centers[0][0], hist_bin_centers[0][-1]]
  min_max_y = [hist_bin_centers[1][0], hist_bin_centers[1][-1]]
  plt.imshow(np.flip(log_density.T, axis=0),
             interpolation='nearest', cmap='Blues')
  plt.xlabel('Values for a random pixel', fontsize=8)
  plt.ylabel('Values for random adjacent pixel', fontsize=8)
  plt.gca().set_xticks([0.0, hist_nbins[0]-1])
  plt.gca().set_yticks([hist_nbins[1]-1, 0.0])
  plt.xlim([-0.5, hist_nbins[0]-0.5])
  plt.ylim([hist_nbins[1]-0.5, -0.5])
  plt.gca().set_xticklabels(['{:.1f}'.format(hist_bin_centers[0][0]),
                             '{:.1f}'.format(hist_bin_centers[0][-1])],
                            fontsize=7)
  plt.gca().set_yticklabels(['{:.1f}'.format(hist_bin_centers[1][0]),
                             '{:.1f}'.format(hist_bin_centers[1][-1])],
                            fontsize=7)
  plt.gca().set_aspect('equal')

  ax = fig.add_subplot(gridspec[1, 3])
  ax.set_title('Joint density of adjacent pixels\nwhitened image', fontsize=10)
  # estimate the joint distribution of any arbitrary pair of adjacent pixels
  num_rand_samps = 10000
  samps = []
  for _ in range(num_rand_samps):
    pix1 = [np.random.randint(low=1, high=w_img.shape[0]-1),
            np.random.randint(low=1, high=w_img.shape[1]-1)]
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1), (0, 1),
               (1, -1), (1, 0), (1, 1)]
    offset = offsets[np.random.choice(np.arange(8))]
    pix2 = (pix1[0] + offset[0], pix1[1] + offset[1])
    samps.append([w_img[pix1[0], pix1[1]], w_img[pix2[0], pix2[1]]])
  samps = np.array(samps)
  hist_nbins = (30, 30)
  counts, hist_bin_edges = np.histogramdd(samps, bins=hist_nbins)
  empirical_density = counts / np.sum(counts)
  log_density = np.copy(empirical_density)
  nonzero_inds = log_density != 0
  log_density[nonzero_inds] = np.log(log_density[nonzero_inds])
  log_density[nonzero_inds] -= np.min(log_density[nonzero_inds])
  log_density[nonzero_inds] /= np.max(log_density[nonzero_inds])
  hist_bin_centers = [(hist_bin_edges[x][:-1] + hist_bin_edges[x][1:]) / 2
                      for x in range(len(hist_bin_edges))]
  min_max_x = [hist_bin_centers[0][0], hist_bin_centers[0][-1]]
  min_max_y = [hist_bin_centers[1][0], hist_bin_centers[1][-1]]
  plt.imshow(np.flip(log_density.T, axis=0),
             interpolation='nearest', cmap='Blues')
  plt.xlabel('Values for a random pixel', fontsize=8)
  plt.ylabel('Values for random adjacent pixel', fontsize=8)
  plt.gca().set_xticks([0.0, hist_nbins[0]-1])
  plt.gca().set_yticks([hist_nbins[1]-1, 0.0])
  plt.xlim([-0.5, hist_nbins[0]-0.5])
  plt.ylim([hist_nbins[1]-0.5, -0.5])
  plt.gca().set_xticklabels(['{:.1f}'.format(hist_bin_centers[0][0]),
                             '{:.1f}'.format(hist_bin_centers[0][-1])],
                            fontsize=7)
  plt.gca().set_yticklabels(['{:.1f}'.format(hist_bin_centers[1][0]),
                             '{:.1f}'.format(hist_bin_centers[1][-1])],
                            fontsize=7)
  plt.gca().set_aspect('equal')

  ax = fig.add_subplot(gridspec[2, 0])
  p_basis_img, p_basis_im_range = plot_utils.get_dictionary_tile_imgs(
      ZCA['PCA_basis'], (8, 8), renormalize=False)
  plt.title('PCA basis used', fontsize=12)
  plt.imshow(p_basis_img[0], cmap='Greys_r', vmin=p_basis_im_range[0][0],
             vmax=p_basis_im_range[0][1])
  plt.axis('off')

  ax = fig.add_subplot(gridspec[2, 1])
  ax.set_title('Variance in each dimension\ncomputed from dset', fontsize=10)
  ax.plot(np.arange(len(ZCA['PCA_axis_variances'])), ZCA['PCA_axis_variances'])
  ax.scatter(np.arange(len(ZCA['PCA_axis_variances'])),
             ZCA['PCA_axis_variances'], s=5)
  ax.set_ylabel('Variance', fontsize=8)
  ax.set_xlabel('PC index', fontsize=8)

  plt.tight_layout()

if __name__ == '__main__':
  main()
