"""
This implements sparse coding dictionary learning
"""
import time
import torch

def train_dictionary(image_dataset, init_dictionary, all_params):
  """
  Train a sparse coding dictionary

  The inputs should vary in their dimensions depending on whether one desires
  fully-connected or convolutional sparse coding. The index ordering convention
  we use is sample index first, in order accomodate PyTorch's indexing
  convention and make the code a bit cleaner. In the fully-connected case this
  differs (by a transpose) compared to how the math is typically written.

  Parameters
  ----------
  image_dataset : torch.Tensor OR torch.Dataloader
      We make __getitem__ calls to either of these iterables and have them
      return us a batch of images. If image_dataset is a torch Tensor, that
      means ALL of the data is stored in the CPU's RAM or in the GPU's RAM. The
      choice of which will have already been made when the Tensor is created.
      ****
      If fully-connected sparse coding is desired, this object has shape
      (k, b, n) where k is the total number of batches, n is the (flattened)
      size of each image, and b is the size of an individual batch. If
      convolutional sparse coding is desired, then this object has shape
      (k, b, c, h, w), where k and b are again the total number of batches and
      the batch size, while c is the number of image channels, h is
      the (padded) height of the image, and w is the (padded) width.
      ****
  init_dictionary : torch.Tensor
      This is an initial guess for the dictionary of basis functions that
      we can use to descibe the images. In the fully-connected case, this is
      torch.Tensor(float32, size=(s, n)) where n is the size of each image and
      s is the size of the code. In the convolutional case, we have
      torch.Tensor(float32, size=(s, c, kh, kw)) where s is the size of the
      code (the number of channels in the resultant code). c is the number
      of image channels and consequently the number of channels for each
      basis function. kh is the kernel height in pixels, while kw is the
      kernel width.
 all_params :
      --- MANDATORY ---
      'mode': str
        One of {'fully-connected', 'convolutional'}
      'num_epochs': int
        The number of times to cycle over the whole dataset, reshuffling the
        order of the patches.
      'code_inference_algorithm' : str
        One of {'ista', 'fista'}
      'dictionary_update_algorithm' : str
        One of {'sc_steepest_descent', 'sc_cheap_quadratic_descent'}
      'inference_param_schedule' : dictionary
        Dictionary containing iteration indexes at which to set/update
        parameters of the inference algorithm. This will be algorithm specific.
        See the docstring for the respective algorithm in analysis_transforms/
      'dict_update_param_schedule' : dictionary
        Dictionary containing iteration indexes at which to set/update
        parameters of the dictionary update algorithm. This will be algorithm
        specific. See the docstring for the respective algorithm in
        dict_update_rules/
      ... IF 'mode' == 'convolutional' ...
      'strides' : tuple(int, int)
        Kernel strides in the vertical and horizontal direction
      'padding' : tuple(tuple(int, int), tuple(int, int)) OR None
        The amount of padding that was done to the images. Vertical, then
        horizontal. Inner index is leading, then trailing padding. If None,
        this is a simple way to indicate that no padding has been done. This is
        preferable to ((0, 0), (0, 0)), which causes undefined behavior.
      --- OPTIONAL ---
      'nonnegative_only' : bool, optional
        Make the codes strictly nonnegative. Default False.
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
    #  this batch. # TODO: calculate this at the beginning if images is not
    #  a DataLoader object
    if coding_mode == 'fully-connected':
      recons = torch.mm(codes, dictionary).cpu().numpy()
      axes_of_summation = 1
    else:
      recons = torch.nn.functional.conv_transpose2d(
          codes, dictionary, stride=kernel_strides).cpu().numpy()
      # get rid of the padding
      if image_padding is not None:
        recons = recons[:, :,
            image_padding[0][0]:-image_padding[0][1],
            image_padding[1][0]:-image_padding[1][1]]
        batch_images_np = batch_images_np[:, :,
            image_padding[0][0]:-image_padding[0][1],
            image_padding[1][0]:-image_padding[1][1]]
      axes_of_summation = (1, 2, 3)
    lasso_l1_component = np.mean(sparsity_weight *
        torch.norm(codes, p=1, dim=axes_of_summation).cpu().numpy())
    lasso_l2_component = np.mean(0.5 *
        np.sum(np.square(recons - batch_images_np), axis=axes_of_summation))
        # for some stupid reason numpy.linalg.norm doesn't work on 4d tensors
    lasso_full = lasso_l2_component + lasso_l1_component
    avg_perc_nonzero = torch.mean(
        torch.norm(codes, p=0, dim=axes_of_summation) /
        np.prod(codes.shape[1:]))
    recon_psnr = []
    for b_idx in range(recons.shape[0]):
      psnr = compute_pSNR(batch_images_np[b_idx],
                          recons[b_idx], manual_sig_mag=batch_sig_mag)
      if psnr != np.inf:
        recon_psnr.append(psnr)
    avg_recon_psnr = np.mean(recon_psnr)
    # Tensorboard doesn't give you a lot of control for how images look so
    # I'm going to generate my own pyplot figures and save these as images.
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

    tb_summary_writer.add_scalar('Average LASSO Loss',
        lasso_full, total_iter_idx)
    tb_summary_writer.add_scalar('Average LASSO L2 component',
        lasso_l2_component, total_iter_idx)
    tb_summary_writer.add_scalar('Average LASSO L1 component',
        lasso_l1_component, total_iter_idx)
    tb_summary_writer.add_scalar('Average Normalized L0',
        avg_perc_nonzero, total_iter_idx)
    tb_summary_writer.add_scalar('Average pSNR of reconstructions',
        avg_recon_psnr, total_iter_idx)

  def write_pyplot_to_tb_image(plt_fig, img_caption):
    buf = io.BytesIO()
    plt_fig.savefig(buf, format='png')
    plt.close(plt_fig)
    the_tensor = torch.tensor(np.array(Image.open(buf))[:, :, :3])
    tb_summary_writer.add_image(img_caption,
        torch.tensor(np.array(Image.open(buf))[:, :, :3]), dataformats='HWC')
  ##########################
  # Done w/ helper functions
  ##########################

  ##########################
  # Setup and error checking
  ##########################
  assert 0 in all_params['inference_param_schedule']
  assert 0 in all_params['dict_update_param_schedule']
  # let's unpack all_params to make things a little less verbose...
  ### MANDATORY ###
  coding_mode = all_params['mode']
  num_epochs = all_params['num_epochs']
  code_inf_alg = all_params['code_inference_algorithm']
  inf_param_schedule = all_params['inference_param_schedule']
  dict_update_alg = all_params['dictionary_update_algorithm']
  dict_update_param_schedule = all_params['dict_update_param_schedule']
  assert coding_mode in ['fully-connected', 'convolutional']
  assert code_inf_alg in ['ista', 'fista']
  assert dict_update_alg in ['sc_steepest_descent',
                             'sc_cheap_quadratic_descent']
  if coding_mode == 'convolutional':
    kernel_strides = all_params['strides']
    image_padding = all_params['padding']
    assert image_padding != ((0, 0), (0, 0)), 'Please use None instead'
  ### OPTIONAL ###
  if 'nonnegative_only' in all_params:
    nonneg_only = all_params['nonnegative_only']
  else:
    nonneg_only = False
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
    if 'reshaped_kernel_size' in trn_vis_sched:
      kernel_reshaping = trn_vis_sched.pop('reshaped_kernel_size')
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

  if code_inf_alg == 'ista':
    if coding_mode == 'fully-connected':
      from analysis_transforms.fully_connected import ista as inference_alg
    else:
      from analysis_transforms.convolutional import ista as inference_alg
  elif code_inf_alg == 'fista':
    if coding_mode == 'fully-connected':
      from analysis_transforms.fully_connected import fista as inference_alg
    else:
      from analysis_transforms.convolutional import fista as inference_alg
  else:
    raise KeyError('Unrecognized code inference algorithm: ' + code_inf_alg)

  if dict_update_alg == 'sc_steepest_descent':
    if coding_mode == 'fully-connected':
      from dict_update_rules.fully_connected import (
          sc_steepest_descent as dict_update)
    else:
      from dict_update_rules.convolutional import (
          sc_steepest_descent as dict_update)
  elif dict_update_alg == 'sc_cheap_quadratic_descent':
    if coding_mode == 'fully-connected':
      from dict_update_rules.fully_connected import (
          sc_cheap_quadratic_descent as dict_update)
    else:
      from dict_update_rules.convolutional import (
          sc_cheap_quadratic_descent as dict_update)
    hessian_diag = init_dictionary.new_zeros(init_dictionary.shape[0])
  else:
    raise KeyError('Unrecognized dict update algorithm: ' + dict_update_alg)

  batch_size = image_dataset[0].shape[0]
  num_batches = len(image_dataset)
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
      if total_iter_idx == 0:
        if batch_images.abs().max() > 10.:
          print('-------\n',
                'Pixel magnitudes are large, consider renormalizing\n-------')
      if total_iter_idx % print_interval == 0:
        print('Iteration', total_iter_idx, 'complete')
        print('Time elapsed:', '{:.1f}'.format(time.time() - starttime),
              'seconds')

      if not batch_images.is_cuda:
        # We have to send image batch to the GPU
        batch_images.cuda()

      ####################
      # Run code inference
      ####################
      # check to see if we need to set/update inference parameters
      if total_iter_idx in inf_param_schedule:
        sparsity_weight = inf_param_schedule[total_iter_idx]['sparsity_weight']
        inf_num_iters = inf_param_schedule[total_iter_idx]['num_iters']

      if coding_mode == 'fully-connected':
        codes = inference_alg.run(batch_images, dictionary, sparsity_weight,
                                  inf_num_iters, nonnegative_only=nonneg_only)
      else:
        codes = inference_alg.run(batch_images, dictionary,
            kernel_stride=kernel_strides, padding_dims=image_padding,
            sparsity_weight=sparsity_weight, num_iters=inf_num_iters,
            nonnegative_only=nonneg_only)

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
      if dict_update_alg == 'sc_steepest_descent':
        if coding_mode == 'fully-connected':
          dict_update.run(batch_images, dictionary, codes,
                          d_upd_stp, d_upd_niters)
        else:
          dict_update.run(batch_images, dictionary, codes,
              kernel_stride=kernel_strides, padding_dims=image_padding,
              stepsize=d_upd_stp, num_iters=d_upd_niters)
      elif dict_update_alg == 'sc_cheap_quadratic_descent':
        if coding_mode == 'fully-connected':
          hessian_diag = (hessian_diag.mul_(0.99) +
                          torch.pow(codes, 2).mean(0)/100)  # credit Yubei Chen
          dict_update.run(batch_images, dictionary, codes, hessian_diag,
                          stepsize=d_upd_stp, num_iters=d_upd_niters)
        else:
          # the diagonal of the hessian has each code correlated with itself
          # which is the sum of squared values:
          hessian_diag = (hessian_diag.mul_(0.99) +
                          torch.mean(torch.sum(codes**2, dim=(2, 3)), dim=0)
                          / 100)
          dict_update.run(batch_images, dictionary, codes, hessian_diag,
              kernel_stride=kernel_strides, padding_dims=image_padding,
              stepsize=d_upd_stp, num_iters=d_upd_niters)

      total_iter_idx += 1

    # we need to reshuffle the batches if we're not using a DataLoader
    if type(image_dataset) == torch.Tensor:
      image_dataset = image_dataset.reshape(
          (-1,) + tuple(image_dataset.shape[2:]))[torch.randperm(
            num_batches * batch_size)].reshape(image_dataset.shape)

    print("Epoch", epoch_idx, "finished")
    # let's make sure we release any unreferenced tensor to make their memory
    # visible to the OS
    torch.cuda.empty_cache() # Sep 17, 2019: This may no longer be needed
