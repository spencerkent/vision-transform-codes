"""
This implements ICA dictionary learning
"""

import time
import os
import pickle
import yaml
from matplotlib import pyplot as plt
import torch

def train_dictionary(image_dataset, init_dictionary, all_params):
  """
  Train an ICA dictionary

  Only works in 'fully-connected' mode

  Parameters
  ----------
  image_dataset : torch.Tensor OR torch.Dataloader
      We make __getitem__ calls to either of these iterables and have them
      return us a batch of images. If image_dataset is a torch Tensor, that
      means ALL of the data is stored in the CPU's RAM or in the GPU's RAM. The
      choice of which will have already been made when the Tensor is created.
      The tensor is an array of size (k, b, n) where k is the total number of
      batches, n is the (flattened) size of each image, and b is the size of
      an individual batch. If image_dataset is a torch.DataLoader that means
      each time we make a __getitem__ call it will return a batch of images
      that it has fetched and preprocessed from disk. This is done in cpu
      multiprocesses that run asynchronously from the GPU. If the whole dataset
      is too large to be loaded into memory, this is really our only option.
  init_dictionary : torch.Tensor(float32, size=(n, n))
      This is an initial guess for the dictionary of basis functions that
      we can use to descibe the images. n is the size of each image and also
      the size of the code -- in ICA the codes are always the same
      dimensionality as the input signal.
 all_params :
      --- MANDATORY ---
      'num_epochs': int
        The number of times to cycle over the whole dataset, reshuffling the
        order of the patches.
      'dictionary_update_algorithm' : str
        One of {'ica_natural_gradient'}
      'dict_update_param_schedule' : dictionary
        Dictionary containing iteration indexes at which to set/update
        parameters of the dictionary update algorithm. This will be algorithm
        specific. See the docstring for the respective algorithm in
        dictionary_learning/
      --- OPTIONAL ---
      'checkpoint_schedule' : dictionary, optional
        Specific iterations at which to save the
        parameters of the model (dictionary, codes, etc.) to disk. Values
        associated w/ each of these keys aren't used and can be set to None.
        We're just using the dictionary for its fast hash-based lookup.
      'training_visualization_schedule' : dictionary, optional
        Specific iterations at which to plot the dictionary and some sample
        codes. Again, dictionary values can be None, we're just using the keys
      'logging_folder_fullpath' : pathlib.Path, optional
        Tells us where to save any checkpoint files or tensorboard summaries.
        Required if either 'checkpoint_schedule' or
        'training_visualization_schedule' is set.
      'stdout_print_interval' : int, optional
        The interval on which we print training progress to the terminal.
        Default 1000.
  """
  ################################
  # Visualization Helper functions
  ################################
  def save_checkpoint(where_to_save):
    # In lieu of the torch-specific saver torch.save, we'll just use
    # pythons's standard serialization tool, pickle. That way we can mess
    # with the results without needing PyTorch.
    pickle.dump(dictionary.cpu().numpy(), open(where_to_save, 'wb'))

  def log_training_progress(current_iteration_number):
    batch_images_np = batch_images.cpu().numpy()
    batch_sig_mag = np.max(batch_images_np) - np.min(batch_images_np)
    #^ psnr depends on the range of the data which we just estimate from
    #  this batch.
    recons = torch.mm(codes, dictionary).cpu().numpy()
    recon_psnr = []
    for b_idx in range(recons.shape[0]):
      psnr = compute_pSNR(batch_images_np[b_idx, :],
                          recons[b_idx, :], manual_sig_mag=batch_sig_mag)
      if psnr != np.inf:
        recon_psnr.append(psnr)
    avg_recon_psnr = np.mean(recon_psnr)
    # Tensorboard doesn't give you a lot of control for how images look so
    # i'm going to generate my own pyplot figures and save these as images.
    # There's probably a more elegant way to do this, but it works for now...
    tiled_kernel_figs = display_dictionary(
        dictionary.cpu().numpy(), reshaping=kernel_reshaping,
        renormalize=True,
        plot_title='Current dictionary (renormalized), iter{}'.format(
          total_iter_idx))
    for fig_idx in range(len(tiled_kernel_figs)):
      tb_img_caption = ('Current dictionary (renorm), fig ' + str(fig_idx+1) +
          ' of ' + str(len(tiled_kernel_figs)))
      write_pyplot_to_tb_image(tiled_kernel_figs[fig_idx], tb_img_caption)
    del tiled_kernel_figs
    tiled_kernel_figs = display_dictionary(
        dictionary.cpu().numpy(), reshaping=kernel_reshaping,
        renormalize=False,
        plot_title='Current dictionary (no renorm), iter {}'.format(
          total_iter_idx))
    for fig_idx in range(len(tiled_kernel_figs)):
      tb_img_caption = ('Current dictionary (no renorm), fig ' +
          str(fig_idx+1) + ' of ' + str(len(tiled_kernel_figs)))
      write_pyplot_to_tb_image(tiled_kernel_figs[fig_idx], tb_img_caption)
    del tiled_kernel_figs

    #TODO: plot the ICA cost function
    tb_summary_writer.add_scalar('Average pSNR of reconstructions',
        avg_recon_psnr, total_iter_idx)

  def write_pyplot_to_tb_image(plt_fig, img_caption):
    buf = io.BytesIO()
    plt_fig.savefig(buf, format='png')
    plt.close(plt_fig)
    the_tensor = torch.tensor(np.array(Image.open(buf))[:, :, :3])
    tb_summary_writer.add_image(img_caption,
        torch.tensor(np.array(Image.open(buf))[:, :, :3]),
        global_step=total_iter_idx, dataformats='HWC')
  ##########################
  # Done w/ helper functions
  ##########################

  ##########################
  # Setup and error checking
  ##########################
  assert 0 in all_params['dict_update_param_schedule']
  assert init_dictionary.size(0) == init_dictionary.size(1) # critically sample
  # let's unpack all_params to make things a little less verbose...
  ### MANDATORY ###
  num_epochs = all_params['num_epochs']
  dict_update_alg = all_params['dictionary_update_algorithm']
  dict_update_param_schedule = all_params['dict_update_param_schedule']
  assert dict_update_alg in ['ica_natural_gradient']
  ### OPTIONAL ###
  if 'logging_folder_fullpath' in all_params:
    assert type(all_params['logging_folder_fullpath']) != str, (
        'should be pathlib.Path')
    logging_path = all_params['logging_folder_fullpath']
    if logging_path.exists() and ('checkpoint_schedule' in all_params or
        'training_visualization_schedule' in all_params):
      print('-------\n',
            'Warning, saving checkpoints and/or tensorboard logs into ',
            'existing, directory. Will overwrite existing files\n-------')
    if not logging_path.exists() and ('checkpoint_schedule' in all_params or
        'training_visualization_schedule' in all_params):
      logging_path.mkdir(parents=True)
  if 'checkpoint_schedule' in all_params:
    import os
    import pickle
    ckpt_sched = all_params['checkpoint_schedule']
  else:
    ckpt_sched = None
  if 'training_visualization_schedule' in all_params:
    import io
    import numpy as np
    from matplotlib import pyplot as plt
    from PIL import Image
    from utils.plotting import compute_pSNR
    from utils.plotting import display_dictionary
    from torch.utils.tensorboard import SummaryWriter
    trn_vis_sched = all_params['training_visualization_schedule']
    tb_summary_writer = SummaryWriter(logging_path)
    if 'reshaped_kernel_size' in all_params:
      kernel_reshaping = all_params.pop('reshaped_kernel_size')
    else:
      kernel_reshaping = None
  else:
    trn_vis_sched = None
  if ckpt_sched is not None or trn_vis_sched is not None:
    import yaml
    # dump the parameters of this training session in human-readable JSON
    saved_training_params = {
        k: all_params[k] for k in all_params if k not in
        ['checkpoint_schedule', 'training_visualization_schedule']}
    yaml.dump(saved_training_params,
              open(logging_path / 'training_params.yaml', 'w'))
  if 'stdout_print_interval' in all_params:
    print_interval = all_params['stdout_print_interval']
  else:
    print_interval = 1000

  from analysis_transforms.fully_connected import invertible_linear
  if dict_update_alg == 'ica_natural_gradient':
    from dict_update_rules.fully_connected import ica_natural_gradient
  else:
    raise KeyError('Unrecognized dict update algorithm: ' + dict_update_alg)
  ##################################
  # Done w/ setup and error checking
  ##################################

  dictionary = init_dictionary  # no copying, just a new reference

  starttime = time.time()
  total_iter_idx = 0
  for epoch_idx in range(num_epochs):
    for batch_idx, batch_images in enumerate(image_dataset):
      ###########################
      # Status updates to console
      ###########################
      if total_iter_idx % print_interval == 0:
        print('Iteration', total_iter_idx, 'complete')
        print('Time elapsed:', '{:.1f}'.format(time.time() - starttime),
              'seconds')
        print('-----')

      if dictionary.device != batch_images.device:
        batch_images = batch_images.to(dictionary.device)

      ####################
      # Run code inference
      ####################
      codes = invertible_linear.run(batch_images, dictionary)

      #################################
      # Checkpointing and visualization
      #################################
      if (ckpt_sched is not None and total_iter_idx in ckpt_sched):
        save_checkpoint(logging_path /
            ('checkpoint_dictionary_iter_' + str(total_iter_idx)))
      if (trn_vis_sched is not None and total_iter_idx in trn_vis_sched):
        log_training_progress(total_iter_idx)

      #######################
      # Update the dictionary
      #######################
      # check to see if we need to set/update parameters
      if total_iter_idx in dict_update_param_schedule:
        d_upd_stp= dict_update_param_schedule[total_iter_idx]['stepsize']
        d_upd_niters = dict_update_param_schedule[total_iter_idx]['num_iters']
      if dict_update_alg == 'ica_natural_gradient':
        ica_natural_gradient.run(dictionary, codes, d_upd_stp, d_upd_niters)

      total_iter_idx += 1

    print("Epoch", epoch_idx, "finished")
