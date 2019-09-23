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
orig_img = unprocessed_images[1]  # arbitrary
dft_num_samples = orig_img.shape
lpf = im_proc.get_low_pass_filter(
    dft_num_samples, {'shape': 'exponential', 'cutoff': 0.1, 'order': 4.0})
lpf_img = im_proc.filter_image(orig_img, lpf)
#^ this is it, the rest is just visualization

# A little visualization
orig_img_recovered = im_proc.filter_image(lpf_img, 1./lpf)
fig = plt.figure(figsize=(15, 6), dpi=100)
plt.subplot(2, 4, 1)
plt.title('Original image', fontsize=12)
plt.imshow(orig_img, cmap='Greys_r', vmin=orig_img.min(), vmax=orig_img.max())
plt.subplot(2, 4, 2)
plt.title('Low-pass filtered image', fontsize=12)
plt.imshow(lpf_img, cmap='Greys_r', vmin=orig_img.min(), vmax=orig_img.max())
plt.xlabel('pSNR to orig: ' + '{:.2f}'.format(
  plot_utils.compute_pSNR(orig_img, lpf_img)) + 'dB')
plt.subplot(2, 4, 3)
plt.title('Recovered Inverted lpf image', fontsize=12)
plt.imshow(orig_img_recovered, cmap='Greys_r',
           vmin=orig_img.min(), vmax=orig_img.max())
plt.xlabel('pSNR to orig: ' + '{:.2f}'.format(
  plot_utils.compute_pSNR(orig_img, orig_img_recovered)) + 'dB')
plt.subplot(2, 4, 4)
plt.title('Difference image', fontsize=12)
plt.imshow(orig_img - orig_img_recovered, cmap='Greys_r',
           vmin=orig_img.min(), vmax=orig_img.max())
filt2dfreqax = plt.subplot(2, 4, 5)
plt.title('Magnitude of filter in 2D Freq', fontsize=12)
imax = plt.imshow(np.fft.fftshift(np.abs(lpf)), cmap='magma',
                  aspect=dft_num_samples[1]/dft_num_samples[0])
divider = make_axes_locatable(filt2dfreqax)
cax = divider.append_axes("right", size="5%", pad=-0.2)
plt.colorbar(imax, cax=cax)
filt2dfreqax.set_yticks([0, dft_num_samples[0]//2, dft_num_samples[0]])
filt2dfreqax.set_xticks([0, dft_num_samples[1]//2, dft_num_samples[1]])
filt2dfreqax.set_yticklabels(['-0.5', '0.0', '0.5'])
filt2dfreqax.set_xticklabels(['-0.5', '0.0', '0.5'])
filt1dv = plt.subplot(2, 4, 6)
plt.title('Slice along vert axis', fontsize=12)
plt.plot(np.linspace(-0.5, 0.5, lpf.shape[0]), np.fft.fftshift(np.abs(lpf[:, 0])))
plt.xlabel('Frequency (units of sampling freq)', fontsize=10)
plt.ylabel('Magnitude of filter', fontsize=10)
filt1dh = plt.subplot(2, 4, 7)
plt.title('Slice along horz axis', fontsize=12)
plt.plot(np.linspace(-0.5, 0.5, lpf.shape[1]), np.fft.fftshift(np.abs(lpf[0, :])))
plt.xlabel('Frequency (units of sampling freq)', fontsize=10)
plt.ylabel('Magnitude of filter', fontsize=10)
filt2dspatax = plt.subplot(2, 4, 8)
plt.title('Filter in image space', fontsize=12)
uncropped = np.fft.fftshift(np.real(np.fft.ifft2(lpf)))
vmiddle = uncropped.shape[0]//2
hmiddle = uncropped.shape[1]//2
cropped = uncropped[vmiddle-20:vmiddle+21, hmiddle-20:hmiddle+21]
imax = plt.imshow(cropped, cmap='Greys_r')
divider = make_axes_locatable(filt2dspatax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(imax, cax=cax)
filt2dspatax.set_yticks([0, 20, 40])
filt2dspatax.set_xticks([0, 20, 40])
filt2dspatax.set_yticklabels(['-20', '0', '20'])
filt2dspatax.set_xticklabels(['-20', '0', '20'])
plt.tight_layout()

plt.show()


###############################################
# Whiten with 'Atick and Redlich whitening, the
# whitening originally used in sparse coding
###############################################
orig_img = unprocessed_images[1]  # arbitrary
dft_num_samples = orig_img.shape
white_img, white_filt = im_proc.whiten_center_surround(
    orig_img, return_filter=True)
orig_img_recovered = im_proc.unwhiten_center_surround(white_img, white_filt)

# A little visualization
fig = plt.figure(figsize=(15, 6), dpi=100)
plt.subplot(2, 4, 1)
plt.title('Original image', fontsize=12)
plt.imshow(orig_img, cmap='Greys_r', vmin=orig_img.min(), vmax=orig_img.max())
plt.subplot(2, 4, 2)
plt.title('Whitened image', fontsize=12)
plt.imshow(white_img, cmap='Greys_r')
plt.xlabel('pSNR to orig: ' + '{:.2f}'.format(
  plot_utils.compute_pSNR(orig_img, white_img)) + 'dB')
plt.subplot(2, 4, 3)
plt.title('Recovered inverse whitened image', fontsize=12)
plt.imshow(orig_img_recovered, cmap='Greys_r',
           vmin=orig_img.min(), vmax=orig_img.max())
plt.xlabel('pSNR to orig: ' + '{:.2f}'.format(
  plot_utils.compute_pSNR(orig_img, orig_img_recovered)) + 'dB')
plt.subplot(2, 4, 4)
plt.title('Difference image', fontsize=12)
plt.imshow(orig_img - orig_img_recovered, cmap='Greys_r',
           vmin=orig_img.min(), vmax=orig_img.max())
filt2dfreqax = plt.subplot(2, 4, 5)
plt.title('Magnitude of filter in 2D Freq', fontsize=12)
imax = plt.imshow(np.fft.fftshift(np.abs(white_filt)), cmap='magma',
                  aspect=dft_num_samples[1]/dft_num_samples[0])
divider = make_axes_locatable(filt2dfreqax)
cax = divider.append_axes("right", size="5%", pad=-0.2)
plt.colorbar(imax, cax=cax)
filt2dfreqax.set_yticks([0, dft_num_samples[0]//2, dft_num_samples[0]])
filt2dfreqax.set_xticks([0, dft_num_samples[1]//2, dft_num_samples[1]])
filt2dfreqax.set_yticklabels(['-0.5', '0.0', '0.5'])
filt2dfreqax.set_xticklabels(['-0.5', '0.0', '0.5'])
filt1dv = plt.subplot(2, 4, 6)
plt.title('Slice along vert axis', fontsize=12)
plt.plot(np.linspace(-0.5, 0.5, white_filt.shape[0]),
         np.fft.fftshift(np.abs(white_filt[:, 0])))
plt.xlabel('Frequency (units of sampling freq)', fontsize=10)
plt.ylabel('Magnitude of filter', fontsize=10)
filt1dh = plt.subplot(2, 4, 7)
plt.title('Slice along horz axis', fontsize=12)
plt.plot(np.linspace(-0.5, 0.5, white_filt.shape[1]),
         np.fft.fftshift(np.abs(white_filt[0, :])))
plt.xlabel('Frequency (units of sampling freq)', fontsize=10)
plt.ylabel('Magnitude of filter', fontsize=10)
filt2dspatax = plt.subplot(2, 4, 8)
plt.title('Filter in image space', fontsize=12)
uncropped = np.fft.fftshift(np.real(np.fft.ifft2(white_filt)))
vmiddle = uncropped.shape[0]//2
hmiddle = uncropped.shape[1]//2
cropped = uncropped[vmiddle-20:vmiddle+21, hmiddle-20:hmiddle+21]
imax = plt.imshow(cropped, cmap='Greys_r')
divider = make_axes_locatable(filt2dspatax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(imax, cax=cax)
filt2dspatax.set_yticks([0, 20, 40])
filt2dspatax.set_xticks([0, 20, 40])
filt2dspatax.set_yticklabels(['-20', '0', '20'])
filt2dspatax.set_xticklabels(['-20', '0', '20'])
plt.tight_layout()

plt.show()
