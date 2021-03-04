"""
Just some simple examples of how to use the image processing utils
"""
import _set_the_path

import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import utils.image_processing as im_proc
import utils.dataset_generation as dset_generation
import utils.plotting as plot_utils
from utils.misc import rotational_average

def main():
  # we'll just show this on Kodak images, you can change this as you see fit
  WHICH_DEMO_IMAGES = 'Kodak_BW'

  if WHICH_DEMO_IMAGES == 'Kodak_BW':
    raw_data_filepath = \
        '/home/spencerkent/Datasets/Kodak/generated_views/train/Grey.p'
    unprocessed_images = pickle.load(open(raw_data_filepath, 'rb'))
    unprocessed_images = [x.astype('float32') for x in unprocessed_images]
  else:
    raise KeyError('not implemented, ' +
                   'see create_patch_training_set in utils/image_processing.py')
  ##########################
  # Low-pass filter an image
  ##########################
  orig_img = unprocessed_images[4]  # arbitrary
  orig_img = orig_img[:, :, None]  # all imgs get a color channel even if grey
  dft_num_samples = orig_img.shape[:2]
  lpf = im_proc.get_low_pass_filter(
      dft_num_samples, {'shape': 'exponential', 'cutoff': 0.1, 'order': 4.0})
  lpf_img = im_proc.filter_fd(orig_img, lpf)
  orig_img_recovered = im_proc.filter_fd(lpf_img, 1./lpf)
  # A little visualization
  visualize_lp_filtering(np.squeeze(orig_img), np.squeeze(lpf_img), lpf,
                         np.squeeze(orig_img_recovered), dft_num_samples)


  ################################################
  # Whiten with 'Atick and Redlich' whitening, the
  # whitening originally used in sparse coding
  ################################################
  dft_num_samples = orig_img.shape[:2]
  white_img, white_filt = im_proc.whiten_center_surround(
      orig_img, cutoffs={'low': 0.00, 'high': 0.8}, return_filter=True)
  orig_img_recovered = im_proc.unwhiten_center_surround(white_img,
      orig_filter_DFT=white_filt)
  # ^exact unwhitening
  visualize_AR_whitening(np.squeeze(orig_img), np.squeeze(white_img),
                         white_filt, np.squeeze(orig_img_recovered),
                         dft_num_samples)

  #############################
  # Whiten with 'ZCA' whitening
  #############################
  # we have to estimate the ZCA_transform on a large batch of data, and we do
  # it for 8x8 patches. Rather than whitening whole images, we just whiten
  # patches and then we can go back and reassemble the image.
  print('Computing ZCA transform...')
  print('Creating a dataset of patches')
  zca_patch_dims = (8, 8)
  one_mil_image_patches = dset_generation.create_patch_training_set(
      num_batches=1, batch_size=1000000, patch_dimensions=zca_patch_dims,
      edge_buffer=5, dataset='Kodak_BW',
      order_of_preproc_ops=['patch'])['batched_patches'][0]
  _, ZCA_params = im_proc.whiten_ZCA(one_mil_image_patches)
  print('Applying transform to test image')
  orig_img_patches, orig_img_patch_pos = im_proc.patches_from_single_image(
      orig_img, zca_patch_dims, flatten_patches=True)
  white_patches = im_proc.whiten_ZCA(orig_img_patches, ZCA_params)
  white_img = im_proc.assemble_image_from_patches(
      white_patches, zca_patch_dims, orig_img_patch_pos)
  orig_img_recovered_patches = im_proc.unwhiten_ZCA(white_patches, ZCA_params)
  orig_img_recovered = im_proc.assemble_image_from_patches(
      orig_img_recovered_patches, zca_patch_dims, orig_img_patch_pos)
  # A little visualization
  visualize_ZCA_whitening(np.squeeze(orig_img), np.squeeze(white_img),
                          ZCA_params, np.squeeze(orig_img_recovered),
                          zca_patch_dims)


  ##############################
  # Local contrast normalization
  ##############################
  normalized_img, normalizer = im_proc.local_contrast_normalization(
      orig_img, 8, return_normalizer=True)
  orig_img_recovered = normalized_img * normalizer
  visualize_lcn(np.squeeze(orig_img), np.squeeze(normalized_img),
                np.squeeze(normalizer), np.squeeze(orig_img_recovered))


  #############################
  # Local luminance subtraction
  #############################
  g_sigma_spatial = 8  # sigma for gaussian filter in spatial domain
  g_sigma_freq = 1. / (2 * np.pi * g_sigma_spatial)
  centered_img, subtractor = im_proc.local_luminance_subtraction(
      orig_img, 8, return_subtractor=True)
  orig_img_recovered = centered_img + subtractor
  visualize_lls(np.squeeze(orig_img), np.squeeze(centered_img),
                np.squeeze(subtractor), np.squeeze(orig_img_recovered),
                g_sigma_freq)


  ######################################
  # This is the type of preprocessing
  # I currently recommend for sparse
  # coding in the context of compression
  ######################################
  # The idea is to pass low-frequencies THROUGH the whitening filter and
  # subtract them out with local luminance subtraction -- we can tune the
  # whitening filter's passband so that it all gets sucked up by the
  # second stage:
  gfilt_sigma_sd = 8  # a gaussian filter with a standard deviation of 8 pix
  wf_cutoff_high = 0.9
  lp_cutoff_attenuation_factor = 100  # calc. freq where power atten. by 100
  gfilt_sigma_fd = 1. / (2 * np.pi * gfilt_sigma_sd)  # in frequency domain
  wf_cutoff_low = (np.sqrt(2 * np.log(np.sqrt(lp_cutoff_attenuation_factor))) *
                   gfilt_sigma_fd)
  dft_num_samples = orig_img.shape[:2]
  white_img, white_filt = im_proc.whiten_center_surround(
      orig_img, cutoffs={'low': wf_cutoff_low, 'high': wf_cutoff_high},
      norm_and_threshold=False, return_filter=True)
  white_centered_img, wc_subtractor = im_proc.local_luminance_subtraction(
      white_img, gfilt_sigma_sd, return_subtractor=True)
  white_img_recovered = white_centered_img + wc_subtractor
  # when we do sparse coding there will be noise that we don't want to
  # accentuate. Therefore, rather than do exact unwhitening, we don't unwhiten
  # the low frequencies.
  orig_img_recovered = im_proc.unwhiten_center_surround(
      white_img_recovered, low_cutoff=wf_cutoff_low)
  visualize_lls(np.squeeze(white_img), np.squeeze(white_centered_img),
                np.squeeze(wc_subtractor), np.squeeze(white_img_recovered),
                g_sigma=gfilt_sigma_fd)
  visualize_AR_whitening(np.squeeze(orig_img), np.squeeze(white_img),
                         white_filt, np.squeeze(orig_img_recovered),
                         dft_num_samples)

  plt.show()


def visualize_lp_filtering(o_img, lp_img, lpf_filt,
                           o_img_recovered, dft_nsamps):
  tall_skinny = o_img.shape[0] > o_img.shape[1]
  if tall_skinny:
    fig = plt.figure(figsize=(15, 15), dpi=100)
    sp_colwidths = [1, 1, 1, 1]
    sp_rowheights = [6, 3, 2]
  else:
    fig = plt.figure(figsize=(15, 8), dpi=100)
    sp_colwidths = [1, 1, 1, 1]
    sp_rowheights = [4, 2, 2]
  fig.suptitle('Low-pass filtering', fontsize=12)

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
  mag = np.abs(np.fft.fft2(o_img, dft_nsamps))
  safe_log10(mag)
  orig_dft_im = ax.imshow(np.fft.fftshift(mag),
      cmap='magma', aspect=dft_nsamps[1]/dft_nsamps[0])
  divider = make_axes_locatable(ax)
  if tall_skinny:
    cax = divider.append_axes("right", size="5%", pad=0.1)
  else:
    cax = divider.append_axes("right", size="5%", pad=-0.2)
  plt.colorbar(orig_dft_im, cax=cax)
  ax.set_yticks([0, dft_nsamps[0]//2, dft_nsamps[0]])
  ax.set_xticks([0, dft_nsamps[1]//2, dft_nsamps[1]])
  ax.set_yticklabels(['-0.5', '0.0', '0.5'])
  ax.set_xticklabels(['-0.5', '0.0', '0.5'])

  ax = fig.add_subplot(gridspec[1, 1])
  ax.set_title('(log) magnitude of 2D DFT\nlow-pass image', fontsize=10)
  mag = np.abs(np.fft.fft2(lp_img, dft_nsamps))
  safe_log10(mag)
  lp_dft_im = ax.imshow(np.fft.fftshift(mag),
      cmap='magma', aspect=dft_nsamps[1]/dft_nsamps[0])
  divider = make_axes_locatable(ax)
  if tall_skinny:
    cax = divider.append_axes("right", size="5%", pad=0.1)
  else:
    cax = divider.append_axes("right", size="5%", pad=-0.2)
  plt.colorbar(lp_dft_im, cax=cax)
  ax.set_yticks([0, dft_nsamps[0]//2, dft_nsamps[0]])
  ax.set_xticks([0, dft_nsamps[1]//2, dft_nsamps[1]])
  ax.set_yticklabels(['-0.5', '0.0', '0.5'])
  ax.set_xticklabels(['-0.5', '0.0', '0.5'])

  ax = fig.add_subplot(gridspec[1, 2])
  ax.set_title('(log) magnitude of 2D DFT\nlow-pass filter', fontsize=10)
  mag = np.abs(lpf_filt)
  safe_log10(mag)
  imax = ax.imshow(np.fft.fftshift(mag), cmap='magma',
                    aspect=dft_nsamps[1]/dft_nsamps[0])
  divider = make_axes_locatable(ax)
  if tall_skinny:
    cax = divider.append_axes("right", size="5%", pad=0.1)
  else:
    cax = divider.append_axes("right", size="5%", pad=-0.2)
  plt.colorbar(imax, cax=cax)
  ax.set_yticks([0, dft_nsamps[0]//2, dft_nsamps[0]])
  ax.set_xticks([0, dft_nsamps[1]//2, dft_nsamps[1]])
  ax.set_yticklabels(['-0.5', '0.0', '0.5'])
  ax.set_xticklabels(['-0.5', '0.0', '0.5'])

  ax = fig.add_subplot(gridspec[1, 3])
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

  ax = fig.add_subplot(gridspec[2, 0])
  ax.set_title('Rotational avg', fontsize=10)
  o_img_fpower = np.abs(np.fft.fft2(o_img))**2
  freq_coords = np.meshgrid(np.fft.fftfreq(o_img.shape[0]),
                            np.fft.fftfreq(o_img.shape[1]), indexing='ij')
  fpower_mean, f = rotational_average(o_img_fpower, 200, freq_coords)
  fpower_mean /= np.max(fpower_mean)
  ax.plot(f, fpower_mean)
  ax.set_ylim([np.min(fpower_mean), np.max(fpower_mean)])
  ax.set_xlim([f[1], f[-1]])
  ax.set_yscale('log')
  ax.set_xscale('log')
  ax.set_xlabel('Frequency (units of sampling freq)', fontsize=8)
  ax.set_ylabel('Normalized signal power', fontsize=8)

  ax = fig.add_subplot(gridspec[2, 1])
  ax.set_title('Rotational avg', fontsize=10)
  lp_img_fpower = np.abs(np.fft.fft2(lp_img))**2
  fpower_mean, f = rotational_average(lp_img_fpower, 200, freq_coords)
  fpower_mean /= np.max(fpower_mean)
  ax.plot(f, fpower_mean)
  ax.set_ylim([np.min(fpower_mean), np.max(fpower_mean)])
  ax.set_xlim([f[1], f[-1]])
  ax.set_yscale('log')
  ax.set_xscale('log')
  ax.set_xlabel('Frequency (units of sampling freq)', fontsize=8)
  ax.set_ylabel('Normalized signal power', fontsize=8)

  ax = fig.add_subplot(gridspec[2, 2])
  ax.set_title('Rotational avg', fontsize=10)
  lp_filt_fpower = np.abs(lpf_filt)**2
  fpower_mean, f = rotational_average(lp_filt_fpower, 200, freq_coords)
  fpower_mean /= np.max(fpower_mean)
  ax.plot(f, fpower_mean)
  ax.set_ylim([np.min(fpower_mean), np.max(fpower_mean)])
  ax.set_xlim([f[1], f[-1]])
  ax.set_yscale('log')
  ax.set_xscale('log')
  ax.set_xlabel('Frequency (units of sampling freq)', fontsize=8)
  ax.set_ylabel('Normalized signal power', fontsize=8)

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
  mag = np.abs(np.fft.fft2(o_img, dft_nsamps))
  safe_log10(mag)
  orig_dft_im = plt.imshow(
      np.fft.fftshift(mag),
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
  mag = np.abs(np.fft.fft2(w_img, dft_nsamps))
  safe_log10(mag)
  orig_dft_im = plt.imshow(
      np.fft.fftshift(mag),
      cmap='magma', aspect=dft_nsamps[1]/dft_nsamps[0])
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=-0.2)
  plt.colorbar(orig_dft_im, cax=cax)
  ax.set_yticks([0, dft_nsamps[0]//2, dft_nsamps[0]])
  ax.set_xticks([0, dft_nsamps[1]//2, dft_nsamps[1]])
  ax.set_yticklabels(['-0.5', '0.0', '0.5'])
  ax.set_xticklabels(['-0.5', '0.0', '0.5'])

  ax = fig.add_subplot(gridspec[1, 2])
  ax.set_title('(log) magnitude of 2D DFT\nwhitening filter', fontsize=10)
  mag = np.abs(w_filt)
  safe_log10(mag)
  imax = plt.imshow(np.fft.fftshift(mag), cmap='magma',
                    aspect=dft_nsamps[1]/dft_nsamps[0])
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=-0.2)
  plt.colorbar(imax, cax=cax)
  ax.set_yticks([0, dft_nsamps[0]//2, dft_nsamps[0]])
  ax.set_xticks([0, dft_nsamps[1]//2, dft_nsamps[1]])
  ax.set_yticklabels(['-0.5', '0.0', '0.5'])
  ax.set_xticklabels(['-0.5', '0.0', '0.5'])

  ax = fig.add_subplot(gridspec[1, 3])
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

  ax = fig.add_subplot(gridspec[2, 0])
  ax.set_title('Rotational avg', fontsize=10)
  o_img_fpower = np.abs(np.fft.fft2(o_img))**2
  freq_coords = np.meshgrid(np.fft.fftfreq(o_img.shape[0]),
                            np.fft.fftfreq(o_img.shape[1]), indexing='ij')
  fpower_mean, f = rotational_average(o_img_fpower, 200, freq_coords)
  fpower_mean /= np.max(fpower_mean)
  ax.plot(f, fpower_mean)
  ax.set_ylim([np.min(fpower_mean), np.max(fpower_mean)])
  ax.set_xlim([f[1], f[-1]])
  ax.set_yscale('log')
  ax.set_xscale('log')
  ax.set_xlabel('Frequency (units of sampling freq)', fontsize=8)
  ax.set_ylabel('Normalized signal power', fontsize=8)

  ax = fig.add_subplot(gridspec[2, 1])
  ax.set_title('Rotational avg', fontsize=10)
  w_img_fpower = np.abs(np.fft.fft2(w_img))**2
  fpower_mean, f = rotational_average(w_img_fpower, 200, freq_coords)
  fpower_mean /= np.max(fpower_mean)
  ax.plot(f, fpower_mean)
  ax.set_ylim([np.min(fpower_mean), np.max(fpower_mean)])
  ax.set_xlim([f[1], f[-1]])
  ax.set_yscale('log')
  ax.set_xscale('log')
  ax.set_xlabel('Frequency (units of sampling freq)', fontsize=8)
  ax.set_ylabel('Normalized signal power', fontsize=8)

  ax = fig.add_subplot(gridspec[2, 2])
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

  ax = fig.add_subplot(gridspec[2, 3])
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

  plt.tight_layout()


def visualize_ZCA_whitening(o_img, w_img, ZCA, o_img_recovered, patch_dims):
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
  p_basis_img = plot_utils.get_dictionary_tile_imgs(
      ZCA['PCA_basis'].T, reshape_to_these_dims=patch_dims)
  plt.title('PCA basis used', fontsize=12)
  plt.imshow(np.squeeze(p_basis_img[0]), cmap='Greys_r')
  plt.axis('off')

  ax = fig.add_subplot(gridspec[1, 3])
  ax.set_title('Variance in each dimension\ncomputed from dset', fontsize=10)
  ax.plot(np.arange(len(ZCA['PCA_axis_variances'])), ZCA['PCA_axis_variances'])
  ax.scatter(np.arange(len(ZCA['PCA_axis_variances'])),
             ZCA['PCA_axis_variances'], s=5)
  ax.set_yscale('log')
  ax.set_ylabel('Variance', fontsize=8)
  ax.set_xlabel('PC index', fontsize=8)

  ax = fig.add_subplot(gridspec[2, 0])
  ax.set_title('Rotational avg', fontsize=10)
  o_img_fpower = np.abs(np.fft.fft2(o_img))**2
  freq_coords = np.meshgrid(np.fft.fftfreq(o_img.shape[0]),
                            np.fft.fftfreq(o_img.shape[1]), indexing='ij')
  fpower_mean, f = rotational_average(o_img_fpower, 200, freq_coords)
  fpower_mean /= np.max(fpower_mean)
  ax.plot(f, fpower_mean)
  ax.set_ylim([np.min(fpower_mean), np.max(fpower_mean)])
  ax.set_xlim([f[1], f[-1]])
  ax.set_yscale('log')
  ax.set_xscale('log')
  ax.set_xlabel('Frequency (units of sampling freq)', fontsize=8)
  ax.set_ylabel('Normalized signal power', fontsize=8)

  ax = fig.add_subplot(gridspec[2, 1])
  ax.set_title('Rotational avg', fontsize=10)
  w_img_fpower = np.abs(np.fft.fft2(w_img))**2
  fpower_mean, f = rotational_average(w_img_fpower, 200, freq_coords)
  fpower_mean /= np.max(fpower_mean)
  ax.plot(f, fpower_mean)
  ax.set_ylim([np.min(fpower_mean), np.max(fpower_mean)])
  ax.set_xlim([f[1], f[-1]])
  ax.set_yscale('log')
  ax.set_xscale('log')
  ax.set_xlabel('Frequency (units of sampling freq)', fontsize=8)
  ax.set_ylabel('Normalized signal power', fontsize=8)

  ax = fig.add_subplot(gridspec[2, 2])
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

  ax = fig.add_subplot(gridspec[2, 3])
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

  plt.tight_layout()

def visualize_lcn(o_img, normed_img, normalizer, o_img_recovered):
  fig = plt.figure(figsize=(15, 8), dpi=100)
  fig.suptitle('Local Contrast Normalization', fontsize=12)
  sp_colwidths = [1, 1, 1, 1]
  sp_rowheights = [4, 2, 2]
  gridspec = fig.add_gridspec(ncols=4, nrows=3, width_ratios=sp_colwidths,
                              height_ratios=sp_rowheights)

  ax = fig.add_subplot(gridspec[0, 0])
  ax.set_title('Original image', fontsize=10)
  ax.imshow(o_img, cmap='Greys_r', vmin=o_img.min(), vmax=o_img.max())

  ax = fig.add_subplot(gridspec[0, 1])
  ax.set_title('Contrast normalized image', fontsize=10)
  ax.imshow(normed_img, cmap='Greys_r', vmin=normed_img.min(),
            vmax=normed_img.max())
  ax.set_xlabel('pSNR to orig: ' + '{:.2f}'.format(
    plot_utils.compute_pSNR(o_img, normed_img)) + 'dB')

  ax = fig.add_subplot(gridspec[0, 2])
  ax.set_title('Divided-out local contrast', fontsize=10)
  ax.imshow(normalizer, cmap='Greys_r', vmin=o_img.min(), vmax=o_img.max())
  ax.set_xlabel('pSNR to orig: ' + '{:.2f}'.format(
    plot_utils.compute_pSNR(o_img, normalizer)) + 'dB')

  ax = fig.add_subplot(gridspec[0, 3])
  ax.set_title('Recovered original image', fontsize=10)
  ax.imshow(o_img_recovered, cmap='Greys_r', vmin=o_img.min(), vmax=o_img.max())
  ax.set_xlabel('pSNR to orig: ' + '{:.2f}'.format(
    plot_utils.compute_pSNR(o_img, o_img_recovered)) + 'dB')

  dft_nsamps = o_img.shape
  ax = fig.add_subplot(gridspec[1, 0])
  ax.set_title('(log) magnitude of 2D DFT\noriginal image', fontsize=10)
  mag = np.abs(np.fft.fft2(o_img, dft_nsamps))
  safe_log10(mag)
  orig_dft_im = plt.imshow(
      np.fft.fftshift(mag),
      cmap='magma', aspect=dft_nsamps[1]/dft_nsamps[0])
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=-0.2)
  plt.colorbar(orig_dft_im, cax=cax)
  ax.set_yticks([0, dft_nsamps[0]//2, dft_nsamps[0]])
  ax.set_xticks([0, dft_nsamps[1]//2, dft_nsamps[1]])
  ax.set_yticklabels(['-0.5', '0.0', '0.5'])
  ax.set_xticklabels(['-0.5', '0.0', '0.5'])

  ax = fig.add_subplot(gridspec[1, 1])
  ax.set_title('(log) magnitude of 2D DFT\ncontrast-normalized image',
               fontsize=10)
  mag = np.abs(np.fft.fft2(normed_img, dft_nsamps))
  safe_log10(mag)
  orig_dft_im = plt.imshow(
      np.fft.fftshift(mag),
      cmap='magma', aspect=dft_nsamps[1]/dft_nsamps[0])
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=-0.2)
  plt.colorbar(orig_dft_im, cax=cax)
  ax.set_yticks([0, dft_nsamps[0]//2, dft_nsamps[0]])
  ax.set_xticks([0, dft_nsamps[1]//2, dft_nsamps[1]])
  ax.set_yticklabels(['-0.5', '0.0', '0.5'])
  ax.set_xticklabels(['-0.5', '0.0', '0.5'])

  ax = fig.add_subplot(gridspec[1, 2])
  ax.set_title('(log) magnitude of 2D DFT\ndivided-out local contrast',
               fontsize=10)
  mag = np.abs(np.fft.fft2(normalizer, dft_nsamps))
  safe_log10(mag)
  orig_dft_im = plt.imshow(
      np.fft.fftshift(mag),
      cmap='magma', aspect=dft_nsamps[1]/dft_nsamps[0])
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=-0.2)
  plt.colorbar(orig_dft_im, cax=cax)
  ax.set_yticks([0, dft_nsamps[0]//2, dft_nsamps[0]])
  ax.set_xticks([0, dft_nsamps[1]//2, dft_nsamps[1]])
  ax.set_yticklabels(['-0.5', '0.0', '0.5'])
  ax.set_xticklabels(['-0.5', '0.0', '0.5'])

  ax = fig.add_subplot(gridspec[2, 0])
  o_img_fpower = np.abs(np.fft.fft2(o_img))**2
  freq_coords = np.meshgrid(np.fft.fftfreq(o_img.shape[0]),
                            np.fft.fftfreq(o_img.shape[1]), indexing='ij')
  fpower_mean, f = rotational_average(o_img_fpower, 200, freq_coords)
  fpower_mean /= np.max(fpower_mean)
  ax.plot(f, fpower_mean)
  ax.set_ylim([np.min(fpower_mean), np.max(fpower_mean)])
  ax.set_xlim([f[1], f[-1]])
  ax.set_yscale('log')
  ax.set_xscale('log')
  ax.set_xlabel('Frequency (units of sampling freq)', fontsize=8)
  ax.set_ylabel('Normalized signal power', fontsize=8)

  ax = fig.add_subplot(gridspec[2, 1])
  n_img_fpower = np.abs(np.fft.fft2(normed_img))**2
  freq_coords = np.meshgrid(np.fft.fftfreq(normed_img.shape[0]),
                            np.fft.fftfreq(normed_img.shape[1]), indexing='ij')
  fpower_mean, f = rotational_average(n_img_fpower, 200, freq_coords)
  fpower_mean /= np.max(fpower_mean)
  ax.plot(f, fpower_mean)
  ax.set_ylim([np.min(fpower_mean), np.max(fpower_mean)])
  ax.set_xlim([f[1], f[-1]])
  ax.set_yscale('log')
  ax.set_xscale('log')
  ax.set_xlabel('Frequency (units of sampling freq)', fontsize=8)
  ax.set_ylabel('Normalized signal power', fontsize=8)

  ax = fig.add_subplot(gridspec[2, 2])
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

  ax = fig.add_subplot(gridspec[2, 3])
  ax.set_title('Joint density of adjacent pixels\ncontrast-normalized image',
               fontsize=10)
  # estimate the joint distribution of any arbitrary pair of adjacent pixels
  num_rand_samps = 10000
  samps = []
  for _ in range(num_rand_samps):
    pix1 = [np.random.randint(low=1, high=normed_img.shape[0]-1),
            np.random.randint(low=1, high=normed_img.shape[1]-1)]
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1), (0, 1),
               (1, -1), (1, 0), (1, 1)]
    offset = offsets[np.random.choice(np.arange(8))]
    pix2 = (pix1[0] + offset[0], pix1[1] + offset[1])
    samps.append([normed_img[pix1[0], pix1[1]], normed_img[pix2[0], pix2[1]]])
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


  plt.tight_layout()


def visualize_lls(o_img, centered_img, subtractor, o_img_recovered, g_sigma):
  fig = plt.figure(figsize=(15, 8), dpi=100)
  fig.suptitle('Local Luminance Subtraction', fontsize=12)
  sp_colwidths = [1, 1, 1, 1]
  sp_rowheights = [4, 2, 2]
  gridspec = fig.add_gridspec(ncols=4, nrows=3, width_ratios=sp_colwidths,
                              height_ratios=sp_rowheights)

  ax = fig.add_subplot(gridspec[0, 0])
  ax.set_title('Original image', fontsize=10)
  ax.imshow(o_img, cmap='Greys_r', vmin=o_img.min(), vmax=o_img.max())

  ax = fig.add_subplot(gridspec[0, 1])
  ax.set_title('Luminance centered image', fontsize=10)
  ax.imshow(centered_img, cmap='Greys_r', vmin=centered_img.min(),
            vmax=centered_img.max())
  ax.set_xlabel('pSNR to orig: ' + '{:.2f}'.format(
    plot_utils.compute_pSNR(o_img, centered_img)) + 'dB')

  ax = fig.add_subplot(gridspec[0, 2])
  ax.set_title('Subtracted-out local luminance', fontsize=10)
  ax.imshow(subtractor, cmap='Greys_r', vmin=o_img.min(), vmax=o_img.max())
  ax.set_xlabel('pSNR to orig: ' + '{:.2f}'.format(
    plot_utils.compute_pSNR(o_img, subtractor)) + 'dB')

  ax = fig.add_subplot(gridspec[0, 3])
  ax.set_title('Recovered original image', fontsize=10)
  ax.imshow(o_img_recovered, cmap='Greys_r', vmin=o_img.min(), vmax=o_img.max())
  ax.set_xlabel('pSNR to orig: ' + '{:.2f}'.format(
    plot_utils.compute_pSNR(o_img, o_img_recovered)) + 'dB')

  dft_nsamps = o_img.shape
  ax = fig.add_subplot(gridspec[1, 0])
  ax.set_title('(log) magnitude of 2D DFT\noriginal image', fontsize=10)
  mag = np.abs(np.fft.fft2(o_img, dft_nsamps))
  safe_log10(mag)
  orig_dft_im = plt.imshow(
      np.fft.fftshift(mag),
      cmap='magma', aspect=dft_nsamps[1]/dft_nsamps[0])
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=-0.2)
  plt.colorbar(orig_dft_im, cax=cax)
  ax.set_yticks([0, dft_nsamps[0]//2, dft_nsamps[0]])
  ax.set_xticks([0, dft_nsamps[1]//2, dft_nsamps[1]])
  ax.set_yticklabels(['-0.5', '0.0', '0.5'])
  ax.set_xticklabels(['-0.5', '0.0', '0.5'])

  ax = fig.add_subplot(gridspec[1, 1])
  ax.set_title('(log) magnitude of 2D DFT\nLuminance-centered image',
               fontsize=10)
  orig_dft_im = plt.imshow(
      np.fft.fftshift(np.log10(np.abs(np.fft.fft2(centered_img, dft_nsamps)))),
      cmap='magma', aspect=dft_nsamps[1]/dft_nsamps[0])
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=-0.2)
  plt.colorbar(orig_dft_im, cax=cax)
  ax.set_yticks([0, dft_nsamps[0]//2, dft_nsamps[0]])
  ax.set_xticks([0, dft_nsamps[1]//2, dft_nsamps[1]])
  ax.set_yticklabels(['-0.5', '0.0', '0.5'])
  ax.set_xticklabels(['-0.5', '0.0', '0.5'])

  ax = fig.add_subplot(gridspec[1, 2])
  ax.set_title('(log) magnitude of 2D DFT\nsubtraced-out local luminance',
               fontsize=10)
  mag = np.abs(np.fft.fft2(subtractor, dft_nsamps))
  safe_log10(mag)
  orig_dft_im = plt.imshow(
      np.fft.fftshift(mag),
      cmap='magma', aspect=dft_nsamps[1]/dft_nsamps[0])
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=-0.2)
  plt.colorbar(orig_dft_im, cax=cax)
  ax.set_yticks([0, dft_nsamps[0]//2, dft_nsamps[0]])
  ax.set_xticks([0, dft_nsamps[1]//2, dft_nsamps[1]])
  ax.set_yticklabels(['-0.5', '0.0', '0.5'])
  ax.set_xticklabels(['-0.5', '0.0', '0.5'])


  ax = fig.add_subplot(gridspec[2, 0])
  o_img_fpower = np.abs(np.fft.fft2(o_img))**2
  freq_coords = np.meshgrid(np.fft.fftfreq(o_img.shape[0]),
                            np.fft.fftfreq(o_img.shape[1]), indexing='ij')
  fpower_mean, f = rotational_average(o_img_fpower, 200, freq_coords)
  fpower_mean /= np.max(fpower_mean)
  ax.plot(f, fpower_mean)
  ax.set_ylim([np.min(fpower_mean), np.max(fpower_mean)])
  ax.set_xlim([f[1], f[-1]])
  ax.set_yscale('log')
  ax.set_xscale('log')
  ax.set_xlabel('Frequency (units of sampling freq)', fontsize=8)
  ax.set_ylabel('Normalized signal power', fontsize=8)

  ax = fig.add_subplot(gridspec[2, 1])
  n_img_fpower = np.abs(np.fft.fft2(centered_img))**2
  freq_coords = np.meshgrid(np.fft.fftfreq(centered_img.shape[0]),
                            np.fft.fftfreq(centered_img.shape[1]), indexing='ij')
  fpower_mean, f = rotational_average(n_img_fpower, 200, freq_coords)
  fpower_mean /= np.max(fpower_mean)
  ax.plot(f, fpower_mean)
  ax.set_ylim([np.min(fpower_mean), np.max(fpower_mean)])
  ax.set_xlim([f[1], f[-1]])
  ax.set_yscale('log')
  ax.set_xscale('log')
  ax.set_xlabel('Frequency (units of sampling freq)', fontsize=8)
  ax.set_ylabel('Normalized signal power', fontsize=8)
  ax.axvline(g_sigma, color='r', linestyle='--')

  ax = fig.add_subplot(gridspec[2, 2])
  n_img_fpower = np.abs(np.fft.fft2(subtractor))**2
  freq_coords = np.meshgrid(np.fft.fftfreq(subtractor.shape[0]),
                            np.fft.fftfreq(subtractor.shape[1]), indexing='ij')
  fpower_mean, f = rotational_average(n_img_fpower, 200, freq_coords)
  fpower_mean /= np.max(fpower_mean)
  ax.plot(f, fpower_mean)
  ax.set_ylim([np.min(fpower_mean), np.max(fpower_mean)])
  ax.set_xlim([f[1], f[-1]])
  ax.set_yscale('log')
  ax.set_xscale('log')
  ax.set_xlabel('Frequency (units of sampling freq)', fontsize=8)
  ax.set_ylabel('Normalized signal power', fontsize=8)
  ax.axvline(g_sigma, color='r', linestyle='--')

  ax = fig.add_subplot(gridspec[1, 3])
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

  ax = fig.add_subplot(gridspec[2, 3])
  ax.set_title('Joint density of adjacent pixels\nLuminance-centered image',
               fontsize=10)
  # estimate the joint distribution of any arbitrary pair of adjacent pixels
  num_rand_samps = 10000
  samps = []
  for _ in range(num_rand_samps):
    pix1 = [np.random.randint(low=1, high=centered_img.shape[0]-1),
            np.random.randint(low=1, high=centered_img.shape[1]-1)]
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1), (0, 1),
               (1, -1), (1, 0), (1, 1)]
    offset = offsets[np.random.choice(np.arange(8))]
    pix2 = (pix1[0] + offset[0], pix1[1] + offset[1])
    samps.append([centered_img[pix1[0], pix1[1]], centered_img[pix2[0], pix2[1]]])
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

  plt.tight_layout()

def safe_log10(nonneg_tensor):
  zeros_inds = (nonneg_tensor == 0)
  nonzeros_inds = (nonneg_tensor > 0)
  nonneg_tensor[nonzeros_inds] = np.log10(nonneg_tensor[nonzeros_inds])
  nonneg_tensor[zeros_inds] = np.min(nonneg_tensor[nonzeros_inds])


if __name__ == '__main__':
  main()
