"""
This implements sparse coding dictionary learning
"""
import time
import torch

from utils import plotting as plot_utils

def train_dictionary(training_image_dataset, validation_image_dataset,
                     init_dictionary, all_params):
  """
  Train a sparse coding dictionary

  The inputs should vary in their dimensions depending on whether one desires
  fully-connected or convolutional sparse coding. The index ordering convention
  we use is to have the sample index first, in order accomodate PyTorch's
  indexing convention for convolutions and to make the code a bit cleaner.
  In the fully-connected case this differs (by a transpose) compared to how
  the math is typically written.

  Parameters
  ----------
  training_image_dataset : torch.Tensor OR torch.Dataloader
      We make __getitem__ calls to either of these iterables and have them
      return us a batch of images.
      ****
      If fully-connected sparse coding is desired, this object has shape
      (k, b, n) where k is the total number of batches, n is the (flattened)
      size of each image, and b is the size of an individual batch. If
      convolutional sparse coding is desired, then this object has shape
      (k, b, c, h, w), where k and b are again the total number of batches and
      the batch size, while c is the number of image channels, h is
      the (padded) height of the image, and w is the (padded) width. If
      using a Dataloader, it is possible that the last batch is smaller than
      the others, if b does not evenly divide the number of total samples.
      This can be avoided by drop_last to True in the DataLoader constructor
      ****
  validation_image_dataset : torch.Tensor OR torch.Dataloader
      Same convention as training images. Often there will be only a single
      large batch, but this gives the flexibility to batch up the validation
      data if it can't fit on the GPU all at once.
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
        One of {'ista', 'fista', 'subspace_ista', 'subspace_fista'}
      'dictionary_update_algorithm' : str
        One of {'sc_steepest_descent', 'sc_cheap_quadratic_descent',
                'subspace_sc_steepest_descent',
                'subspace_sc_cheap_quadratic_descent'}
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
      ... IF 'code_inference_algorithm' or 'dictionary_update_algorithm' are
          prefixed by 'subspace' ...
      This means we're doing subspace variants of inference, learning, or (in
      all likelihood) both. We need a couple additional parameters:
      'group_assignments' : list(array_like)
        Identifies a grouping of the dictionary elements--used by the
        subspace inference and dictionary update algs. Our convention is the
        following: Suppose we have
          group_assignments = [[0, 2, 5], [1], [2, 3, 4, 5]]
        This specifies three groups. group 0 is comprised of elements 0, 2,
        and 5 from the dictioanary, group 1 is composed of element 1, and
        group 2 is composed of elements 2, 3, 4, and 5. Notice that each group
        can be of a different size and elements of the dictionary can
        participate in multiple groups.
      'subspace_alignment_penalty' : float
        We impose a regularization penalty on the alignment of dictionary
        elements that are part of the same group. Without this, dictionary
        updates tend to produce some duplicated elements within a subgroup.
        This is the lagrange multiplier beta, which weights this penalty in the
        full loss function.
      --- OPTIONAL ---
      'nonnegative_only' : bool, optional
        Make the codes strictly nonnegative. Default False.
      'checkpoint_schedule' : set, optional
        Specific iterations at which to save the
        parameters of the model (dictionary, codes, etc.) to disk.
      'training_visualization_schedule' : set, optional
        Specific iterations at which to plot the dictionary and some sample
        codes.
      'logging_folder_fullpath' : pathlib.Path, optional
        Tells us where to save any checkpoint files or tensorboard summaries.
        Required if either 'checkpoint_schedule' or
        'training_visualization_schedule' is provided.
      'stdout_print_interval' : int, optional
        The interval on which we print training progress to the terminal.
        Default 1000.
  """
  #######################
  # Misc Helper functions
  #######################
  # I realize there's a fair amount of variable scope flying around here, but
  # for now I think this is the best way to limit the verbosity of the trainer
  def infer_codes(batch_images):
    """Infers sparse codes for a batch of images"""
    inf_alg_inputs = {
        'dictionary': dictionary, 'sparsity_weight': sparsity_weight,
        'num_iters': inf_num_iters, 'nonnegative_only': nonneg_only,
        'hard_threshold': hard_threshold}
    if coding_mode == 'fully-connected':
      inf_alg_inputs.update({'images': batch_images})
    else:
      inf_alg_inputs.update({'images_padded': batch_images,
        'kernel_stride': kernel_strides, 'padding_dims': image_padding})
    if code_inf_alg in ['subspace_ista', 'subspace_fista']:
      inf_alg_inputs.update({'group_assignments': group_assignments})
      inf_alg_inputs.pop('nonnegative_only')  # these vers. always nonnegative
    batch_codes = inference_alg.run(**inf_alg_inputs)
    return batch_codes

  def update_dictionary(batch_images, batch_codes):
    """Updates the dictionary based on a batch of sparse codes"""
    dict_upd_alg_inputs = {'dictionary': dictionary, 'codes': batch_codes,
      'stepsize': d_upd_stp, 'num_iters': d_upd_niters}
    if coding_mode == 'fully-connected':
      dict_upd_alg_inputs.update({'images': batch_images})
    else:
      dict_upd_alg_inputs.update({'images_padded': batch_images,
        'kernel_stride': kernel_strides, 'padding_dims': image_padding})
    if dict_update_alg in ['sc_cheap_quadratic_descent',
                           'subspace_sc_cheap_quadratic_descent']:
      if coding_mode == 'fully-connected':
        hessian_diag.mul_(0.99).add_(torch.pow(batch_codes, 2).mean(0)/100)
      else:
        if dict_update_alg == 'subspace_sc_cheap_quadratic_descent':
          raise NotImplementedError('TODO for convolutional')
        # the diagonal of the hessian has each code correlated with itself
        # which is the sum of squared values:
        hessian_diag.mul_(0.99).add_(
            torch.mean(torch.sum(batch_codes**2, dim=(2, 3)), dim=0) / 100)
      dict_upd_alg_inputs.update({'hessian_diagonal': hessian_diag})
    if dict_update_alg in ['subspace_sc_steepest_descent',
                           'subspace_sc_cheap_quadratic_descent']:
      dict_upd_alg_inputs.update({
        'group_assignments': group_assignments,
        'alignment_penalty': subspace_alignment_penalty})
    dict_update.run(**dict_upd_alg_inputs)

  def save_checkpoint(where_to_save):
    """Checkpoints the current dictionary to disk"""
    # In lieu of the torch-specific saver torch.save, we'll just use
    # pythons's standard serialization tool, pickle. That way we can mess
    # with the results without needing PyTorch.
    pickle.dump(dictionary.cpu().numpy(), open(where_to_save, 'wb'))

  def compute_metrics(batch_images, batch_codes):
    """These metrics can be used to assess how training is progressing"""
    metrics = {}
    batch_images_np = batch_images.cpu().numpy()
    if coding_mode == 'fully-connected':
      recons = torch.mm(batch_codes, dictionary).cpu().numpy()
      axes_of_summation = 1
    else:
      recons = torch.nn.functional.conv_transpose2d(
          batch_codes, dictionary, stride=kernel_strides).cpu().numpy()
      # get rid of the padding
      if image_padding is not None:
        recons = recons[:, :,
            image_padding[0][0]:-image_padding[0][1],
            image_padding[1][0]:-image_padding[1][1]]
        batch_images_np = batch_images_np[:, :,
            image_padding[0][0]:-image_padding[0][1],
            image_padding[1][0]:-image_padding[1][1]]
      axes_of_summation = (1, 2, 3)
    metrics['Average LASSO L2 component'] = np.mean(0.5 *
        np.sum(np.square(recons - batch_images_np), axis=axes_of_summation))
        # for some stupid reason numpy.linalg.norm doesn't work on 4d tensors
    if code_inf_alg in ['subspace_ista', 'subspace_fista']:
      sum_of_group_norms = np.zeros((len(batch_codes),))
      for g_idx in range(len(group_assignments)):
        sum_of_group_norms += torch.norm(
            batch_codes[:, group_assignments[g_idx]], p=2, dim=1).cpu().numpy()
      metrics['Average LASSO lagrange component'] = np.mean(
          sparsity_weight * sum_of_group_norms)
      # for now I won't include the alignment regularization because it's
      # a little involved to compute, but it could be added later.
    else:
      metrics['Average LASSO lagrange component'] = np.mean(sparsity_weight *
          torch.norm(batch_codes, p=1, dim=axes_of_summation).cpu().numpy())
    metrics['Average LASSO Loss'] = (
        metrics['Average LASSO L2 component'] +
        metrics['Average LASSO lagrange component'])
    metrics['Average Normalized L0'] = float(torch.mean(
        torch.norm(batch_codes, p=0, dim=axes_of_summation) /
        np.prod(batch_codes.shape[1:])).cpu().numpy())
    batch_sig_mag = np.max(batch_images_np) - np.min(batch_images_np)
    #^ psnr depends on the range of the data, which we estimate from batch.
    recon_psnr = []
    for b_idx in range(recons.shape[0]):
      psnr = compute_pSNR(batch_images_np[b_idx], recons[b_idx],
                          manual_sig_mag=batch_sig_mag)
      if psnr != np.inf:
        recon_psnr.append(psnr)
    metrics['Average pSNR of reconstructions'] = np.mean(recon_psnr)
    metrics['Average change in dictionary kernels'] = torch.mean(
        torch.abs(dictionary - previous_dictionary),
        dim=axes_of_summation).cpu().numpy()
    return metrics

  def send_metrics_to_tensorboard(dict_of_metrics):
    """Used in tensorboard visualization of training progress"""
    for metric in dict_of_metrics:
      tb_summary_writer.add_scalar(
          metric, dict_of_metrics[metric], total_iter_idx)

  def send_dict_viz_to_tensorboard():
    """Writes a image summary to tensorboard for nice visualization of dict"""
    # Tensorboard doesn't give you a lot of control for how images look so
    # I'm going to generate my own pyplot figures and save these as images.
    # There's probably a more elegant way to do this, but it works for now...
    tiled_kernel_figs = display_dictionary(
        dictionary.cpu().numpy(), reshaping=kernel_reshaping,
        groupings=group_assignments, renormalize=True,
        plot_title='Current dictionary (renormalized), iter{}'.format(
          total_iter_idx))
    for fig_idx in range(len(tiled_kernel_figs)):
      tb_img_caption = ('Current dictionary (renorm), fig ' + str(fig_idx+1) +
          ' of ' + str(len(tiled_kernel_figs)))
      write_pyplot_to_tb_image(tiled_kernel_figs[fig_idx], tb_img_caption)
    del tiled_kernel_figs
    tiled_kernel_figs = display_dictionary(
        dictionary.cpu().numpy(), reshaping=kernel_reshaping,
        groupings=group_assignments, renormalize=False,
        plot_title='Current dictionary (no renorm), iter {}'.format(
          total_iter_idx))
    for fig_idx in range(len(tiled_kernel_figs)):
      tb_img_caption = ('Current dictionary (no renorm), fig ' +
          str(fig_idx+1) + ' of ' + str(len(tiled_kernel_figs)))
      write_pyplot_to_tb_image(tiled_kernel_figs[fig_idx], tb_img_caption)
    del tiled_kernel_figs

  def write_pyplot_to_tb_image(plt_fig, img_caption):
    """This is the sort of hacky way we get pyplot figures saved to TB"""
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
  assert code_inf_alg in ['ista', 'fista', 'subspace_ista', 'subspace_fista']
  assert dict_update_alg in ['sc_steepest_descent',
                             'sc_cheap_quadratic_descent',
                             'subspace_sc_steepest_descent',
                             'subspace_sc_cheap_quadratic_descent']
  if coding_mode == 'convolutional':
    kernel_strides = all_params['strides']
    image_padding = all_params['padding']
    assert image_padding != ((0, 0), (0, 0)), 'Please use None instead'
  ### OPTIONAL ###
  if 'nonnegative_only' in all_params:
    nonneg_only = all_params['nonnegative_only']
  else:
    nonneg_only = False
  if 'hard_threshold' in all_params:
    hard_threshold = all_params['hard_threshold']
  else:
    hard_threshold = False
  if 'group_assignments' in all_params:
    # specify each group's members once, no duplicates
    assert all([len(set(x)) == len(x) for x in all_params['group_assignments']])
    group_assignments = all_params['group_assignments']
    if type(group_assignments[0]) != list:
      # yaml serialization of numpy arrays doesn't work well, so we'll make
      # sure it's a list, just in case.
      group_assignments = [x.tolist() for x in group_assignments]
  else:
    group_assignments = None
  if 'renormalize_dictionary' in all_params:
    renormalize_dictionary = all_params['renormalize_dictionary']
  else:
    renormalize_dictionary = True
  if renormalize_dictionary:
    assert torch.allclose(init_dictionary.norm(p=2, dim=1),
                          torch.tensor(1.).to(init_dictionary.device)), (
           'Please ensure the initial dictionary is already normalized')
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
      kernel_reshaping = all_params['reshaped_kernel_size']
    else:
      kernel_reshaping = None
  else:
    trn_vis_sched = None
  if ckpt_sched is not None or trn_vis_sched is not None:
    import yaml
    saved_training_params = {
        k: all_params[k] for k in all_params if k not in
        ['checkpoint_schedule', 'training_visualization_schedule',
         'group_assignments']}
    # add back in list-ified group assignment
    saved_training_params.update({'group_assignments': group_assignments})
    yaml.dump(saved_training_params,
              open(logging_path / 'training_params.yaml', 'w'),
              default_flow_style=None)
    # for good measure, let's just save the entire calling script. We have it
    # as a string here:
    with open(logging_path / 'called_script.py', 'w') as saved_script:
      saved_script.write(all_params['str_entire_calling_script'])
  if 'stdout_print_interval' in all_params:
    print_interval = all_params['stdout_print_interval']
  else:
    print_interval = 1000
  if 'dict_element_rp_schedule' in all_params:  # {r}eset or {p}rune schedule
    dict_element_rp_schedule = all_params['dict_element_rp_schedule']
  else:
    dict_element_rp_schedule = None

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
  elif code_inf_alg in ['subspace_ista', 'subspace_fista']:
    assert group_assignments is not None
    if coding_mode == 'fully-connected':
      if code_inf_alg == 'subspace_ista':
        from analysis_transforms.fully_connected import (
            subspace_ista as inference_alg)
      else:
        from analysis_transforms.fully_connected import (
            subspace_fista as inference_alg)
    else:
      raise KeyError('Havent implemented subspace ISTA for convolutional yet')
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
  elif dict_update_alg == 'subspace_sc_steepest_descent':
    if coding_mode == 'fully-connected':
      from dict_update_rules.fully_connected import (
          subspace_sc_steepest_descent as dict_update)
    else:
      raise KeyError('Not implemented for convolutional')
    group_assignments = all_params['group_assignments']
    subspace_alignment_penalty = all_params['subspace_alignment_penalty']
  elif dict_update_alg == 'subspace_sc_cheap_quadratic_descent':
    if coding_mode == 'fully-connected':
      from dict_update_rules.fully_connected import (
          subspace_sc_cheap_quadratic_descent as dict_update)
    else:
      raise KeyError('Not implemented for convolutional')
    group_assignments = all_params['group_assignments']
    subspace_alignment_penalty = all_params['subspace_alignment_penalty']
    hessian_diag = init_dictionary.new_zeros(init_dictionary.shape[0])
  else:
    raise KeyError('Unrecognized dict update algorithm: ' + dict_update_alg)
  ##################################
  # Done w/ setup and error checking
  ##################################

  dictionary = init_dictionary  # no copying, just a new reference
  previous_dictionary = torch.zeros_like(dictionary)
  previous_dictionary.copy_(dictionary)

  starttime = time.time()
  total_iter_idx = 0
  for epoch_idx in range(num_epochs):
    for t_batch_idx, t_batch_images in enumerate(training_image_dataset):

      if total_iter_idx % print_interval == 0 and total_iter_idx != 0:
        print(total_iter_idx, 'iterations complete')
        print('Time elapsed:', '{:.1f}'.format(time.time() - starttime),
              'seconds')
        print('-----')

      # setting parameters of inference and dictionary update
      if total_iter_idx in inf_param_schedule:
        sparsity_weight = inf_param_schedule[total_iter_idx]['sparsity_weight']
        inf_num_iters = inf_param_schedule[total_iter_idx]['num_iters']
      if total_iter_idx in dict_update_param_schedule:
        d_upd_stp= dict_update_param_schedule[total_iter_idx]['stepsize']
        d_upd_niters = dict_update_param_schedule[total_iter_idx]['num_iters']
      # {r}eseting or {p}runing dictionary elements, if necessary
      if (dict_element_rp_schedule is not None and
          total_iter_idx in dict_element_rp_schedule):
        f_type = dict_element_rp_schedule[total_iter_idx]['filter_type']
        f_params = dict_element_rp_schedule[total_iter_idx]['filter_params']
        f_params.update({'group_assignments': group_assignments,
                         'coding_mode': coding_mode})
        rp_action = dict_element_rp_schedule[total_iter_idx]['action']
        # at least one of the reset-or-prune modes requires an estimate
        # of the code distribution--we'll use the validation dataset for this
        v_codes = []
        for v_batch_images in validation_image_dataset:
          if dictionary.device != v_batch_images.device:
            v_batch_images = v_batch_images.to(dictionary.device)
          v_codes.append(infer_codes(v_batch_images))
        v_codes = torch.cat(v_codes)  # whole validation set...
        dictionary, affected_elems = reset_or_prune_dict_elements(
            dictionary, v_codes, f_type, f_params, rp_action)
        if rp_action == 'prune' and len(affected_elems) > 0:
          previous_dictionary.copy_(dictionary)
          if dict_update_alg in ['sc_cheap_quadratic_descent',
                                 'subspace_sc_cheap_quadratic_descent']:
            inds = torch.ones(dictionary.size(0), dtype=torch.bool)
            inds[affected_elems] = False
            hessian_diag = hessian_diag[inds]
          # TODO: WARNING: doesn't yet work for subspaces.
          # (have to update group element inds)
      # checkpointing the dictionary
      if (ckpt_sched is not None and total_iter_idx in ckpt_sched):
        save_checkpoint(logging_path /
            ('checkpoint_dictionary_iter_' + str(total_iter_idx)))
      # writing a training visualization log
      if (trn_vis_sched is not None and total_iter_idx in trn_vis_sched):
        val_metrics = []
        for v_batch_images in validation_image_dataset:
          if dictionary.device != v_batch_images.device:
            v_batch_images = v_batch_images.to(dictionary.device)
          v_codes = infer_codes(v_batch_images)
          val_metrics.append(compute_metrics(v_batch_images, v_codes))
        send_metrics_to_tensorboard({x: np.mean([val_metrics[y][x]
          for y in range(len(val_metrics))]) for x in val_metrics[0]})
        send_dict_viz_to_tensorboard()

      # this is what gets executed every loop--code inference followed by
      # dictionary update
      if dictionary.device != t_batch_images.device:
        t_batch_images = t_batch_images.to(dictionary.device)
      t_codes = infer_codes(t_batch_images)
      previous_dictionary.copy_(dictionary)
      update_dictionary(t_batch_images, t_codes)

      total_iter_idx += 1

    print("Epoch", epoch_idx + 1, "finished")


def reset_or_prune_dict_elements(dictionary, codes, filter_type,
                                 filter_params, action):
  """
  Reset (to random) or prune some elements of a dictionary used during training

  I'm defining this outside the train_dictionary() function because it's fairly
  self-contained and otherwise would be an annoyingly long helper function

  Parameters
  ----------
  dictionary : see docstring for train_dictionary
  codes : see docstring for train_dictionary
  filter_type : string
      One of {'random', 'cosine_sim_threshold', 'nonuniformity_within_group'}
      'cosine_sim_threshold' looks for pairs of dictionary elements whose
      cosine similarity is below a threshold (either selected by user at
      runtime provided by the filter_parameters input).
      'nonuniformity_within_group' estimates the degree of uniformity for the
      density of grouped code phases. Will flag groups which don't uniformly
      file the hypersphere. 'random' is self explanatory.
  filter_params : dictionary
      'coding_mode' : one of {'fully-connected', 'convolutional'}
      'group_assignments' : see docstring for train_dictionary. Can be None.
      ... IF filter_type == 'random' ...
      'num_to_modify' : int. The number of dictionary elements to randomly mod
      ... IF filter_type == 'cosine_sim_threshold' ...
      'cue_user' : bool. Ask the user what the threshold should be (pop up)
      'only_sim_within_group' : bool. Only compute cosine sim within group
      ... IF filter_type == 'cosine_sim_threshold' AND 'cue_user' == False...
      'threshold' : float. The threshold of cosine similarity to use
      ... IF filter_type == 'nonuniformity_within_group' ...
      'num_gc_in_average' : int. Number of great circles to use for estimating
        uniformity
  action : str
      One of {'reset', 'prune'}. Pretty self-explanatory
  """
  import numpy as np
  from matplotlib import pyplot as plt

  groups = filter_params['group_assignments']
  coding_mode = filter_params['coding_mode']
  if coding_mode == 'fully-connected':
    if filter_type == 'random':
      modify_these = np.random.choice(np.arange(dictionary.size(0)),
                                      filter_params['num_to_modify'])
      if action == 'reset':
        # reset with noise that has norm equal to the average norm across dict
        average_norm = torch.mean(dictionary.norm(p=2, dim=1))
        noise = torch.randn((len(modify_these), dictionary.size(1)),
                            device=dictionary.device)
        noise.mul_(average_norm / noise.norm(p=2, dim=1)[:, None])
        dictionary[modify_these] = noise
      else:
        # prune these
        inds = torch.ones(dictionary.size(0), dtype=torch.bool)
        inds[modify_these] = False
        dictionary = dictionary[inds]
        for g_idx in range(len(groups)):
          new_group = []
          for de_idx in groups[g_idx]:
            if de_idx not in modify_these:
              new_group.append(de_idx)
          groups[g_idx] = new_group
      return dictionary, modify_these

    elif filter_type == 'cosine_sim_threshold':
      # reset element that are too close in cosine similarity to other elements
      if filter_params['cue_user']:
        # ask the user what the threshold should be
        if filter_params['only_sim_within_group']:
          # do this one group at a time, only use similarity within group
          assert groups is not None
          all_similarities = []
          for g_idx in range(len(groups)):
            norms = torch.norm(dictionary[groups[g_idx]],
                               p=2, dim=1, keepdim=True)
            cos_sims = (torch.mm(dictionary[groups[g_idx]],
                                 dictionary[groups[g_idx]].t()) /
                        torch.mm(norms, norms.t())).cpu().numpy()
            upper_diag_inds = np.triu_indices(
                n=cos_sims.shape[0], k=1, m=cos_sims.shape[1])
            upper_diag_sims_flat = cos_sims[upper_diag_inds[0],
                                            upper_diag_inds[1]]
            all_similarities.append(upper_diag_sims_flat)
          all_similarities = np.concatenate(all_similarities)
        else:
          norms = torch.norm(dictionary, p=2, dim=1, keepdim=True)
          cos_sims = (torch.mm(dictionary, dictionary.t()) /
                      torch.mm(norms, norms.t())).cpu().numpy()
          upper_diag_inds = np.triu_indices(
              n=cos_sims.shape[0], k=1, m=cos_sims.shape[1])
          all_similarities = cos_sims[upper_diag_inds[0], upper_diag_inds[1]]
        sim_fig = plt.figure()
        ax = plt.subplot(111)
        ax.bar(np.arange(len(all_similarities)), all_similarities,
               align='center', color='b', width=1)
        ax.set_ylabel('Pairwise cosine similarities')
        ax.set_xlabel('Possible distinct pairs')
        plt.show()
        csim_threshold = float(input(
          'Please select a threshold for resetting/pruning dict elements: '))
      else:
        # don't cue user, this should be a parameter
        csim_threshold = filter_params['threshold']
      if filter_params['only_sim_within_group']:
        modify_these = []
        for g_idx in range(len(groups)):
          norms = torch.norm(dictionary[groups[g_idx]],
                             p=2, dim=1, keepdim=True)
          cos_sims = (torch.mm(dictionary[groups[g_idx]],
                               dictionary[groups[g_idx]].t()) /
                      torch.mm(norms, norms.t())).cpu().numpy()
          problem_pairs = np.argwhere(np.abs(np.triu(cos_sims, k=1)) > 
                                      csim_threshold)
          temp_mt = []
          for pair in problem_pairs:
            if pair[0] not in temp_mt and pair[1] not in temp_mt:
              temp_mt.append(pair[np.random.choice([0, 1])])
          if len(temp_mt) > 0:
            print('Action ', action, 'applied to ', temp_mt, 'in group', g_idx)
            if action == 'reset':
              average_norm = torch.mean(
                  dictionary[groups[g_idx]].norm(p=2, dim=1))
              noise = torch.randn((len(temp_mt), dictionary.shape[1]),
                                  device=dictionary.device)
              noise.mul_(average_norm / noise.norm(p=2, dim=1)[:, None])
              dictionary[np.array(groups[g_idx])[temp_mt]] = noise
            modify_these.append(np.array(groups[g_idx])[temp_mt])
        modify_these = np.array(modify_these).flatten()
        if action == 'prune':
          if len(modify_these) > 0:
            # prune
            inds = torch.ones(dictionary.size(0), dtype=torch.bool)
            inds[modify_these] = False
            dictionary = dictionary[inds]
            for g_idx in range(len(groups)):
              new_group = []
              for de_idx in groups[g_idx]:
                if de_idx not in modify_these:
                  new_group.append(de_idx)
              groups[g_idx] = new_group
        return dictionary, modify_these

      else:  # not restricted to groups
        norms = torch.norm(dictionary, p=2, dim=1, keepdim=True)
        cos_sims = (torch.mm(dictionary, dictionary.t()) /
                    torch.mm(norms, norms.t())).cpu().numpy()
        problem_pairs = np.argwhere(np.triu(cos_sims, k=1) > csim_threshold)
        modify_these = []
        for pair in problem_pairs:
          if pair[0] not in modify_these and pair[1] not in modify_these:
            modify_these.append(pair[np.random.choice([0, 1])])
        modify_these = np.array(modify_these)
        if len(modify_these) > 0:
          if action == 'reset':
            average_norm = torch.mean(dictionary.norm(p=2, dim=1))
            noise = torch.randn((len(modify_these), dictionary.shape[1]),
                                device=dictionary.device)
            noise.mul_(average_norm / noise.norm(p=2, dim=1)[:, None])
            dictionary[modify_these] = noise
          else:
            # prune
            inds = torch.ones(dictionary.size(0), dtype=torch.bool)
            inds[modify_these] = False
            dictionary = dictionary[inds]
            for g_idx in range(len(groups)):
              new_group = []
              for de_idx in groups[g_idx]:
                if de_idx not in modify_these:
                  new_group.append(de_idx)
              groups[g_idx] = new_group
        return dictionary, modify_these

    elif filter_type == 'nonuniformity_within_group':
      # identify groups that are outliers in the degree to which nonzero
      # activations are nonuniformly distributed within the group.
      num_great_circles = filter_params['num_gc_in_average']
      group_renormed_mean_variances = []
      for g_idx in range(len(groups)):
        samples_where_nonzero = torch.nonzero(
            torch.sum(codes[: , groups[g_idx]] != 0, dim=1), as_tuple=True)[0]
        nz_codes = codes[samples_where_nonzero][:,  groups[g_idx]]
        # removed_norm
        norms = torch.norm(nz_codes, p=2, dim=1, keepdim=True)
        renormed = nz_codes / norms
        # we take projections onto *great circles* of the unit sphere,
        # computing the average variance of these projections as a proxy
        # measure of the amount of nonuniformity for the renormalized codes
        variances = []
        for gc_idx in range(num_great_circles):
          # get a random great circle of the unit sphere
          first_vec = np.random.randn(len(groups[g_idx]))
          first_vec /= np.linalg.norm(first_vec)
          second_vec = np.random.randn(len(groups[g_idx]))
          second_vec /= np.linalg.norm(second_vec)
          plane_basis, _ = np.linalg.qr(np.c_[first_vec, second_vec])
          great_circle_projection = np.dot(renormed.cpu().numpy(), plane_basis)
          projection_angle = np.angle(great_circle_projection[:, 0] +
                                      1j*great_circle_projection[:, 1])
          hist_nbins = 20
          histogram_bin_edges = np.linspace(-np.pi, np.pi, hist_nbins+1)
          histogram_bin_centers = (histogram_bin_edges[:-1] +
                                   histogram_bin_edges[1:]) / 2
          counts, _ = np.histogram(projection_angle, histogram_bin_edges)
          empirical_density = counts / np.sum(counts)
          variances.append(np.var(empirical_density))
        group_renormed_mean_variances.append(np.mean(variances))
      # we look at all the groups and flag any group who's mean variance
      # is more than 1 standard deviation from the mean of these means
      mean_of_means = np.mean(group_renormed_mean_variances)
      std_of_means = np.std(group_renormed_mean_variances)
      g_modify_these = np.nonzero(
          np.logical_and(
            np.abs(np.array(group_renormed_mean_variances) - mean_of_means) >
            std_of_means,
            np.abs(np.array(group_renormed_mean_variances)) > 0.002))[0]
      modify_these = np.array([groups[x] for x in g_modify_these]).flatten()
      # reset with noise that has norm equal to the average norm across dict
      if len(modify_these) > 0:
        if action == 'reset':
          average_norm = torch.mean(dictionary.norm(p=2, dim=1))
          noise = torch.randn((len(modify_these), dictionary.shape[1]),
                              device=dictionary.device)
          noise.mul_(average_norm / noise.norm(p=2, dim=1)[:, None])
          dictionary[modify_these] = noise
        else:
          # prune, not exactly sure which to prune here. I'll just pick one
          inds = torch.ones(dictionary.size(0), dtype=torch.bool)
          inds[modify_these] = False
          dictionary = dictionary[inds]
          for g_idx in range(len(groups)):
            new_group = []
            for de_idx in groups[g_idx]:
              if de_idx not in modify_these:
                new_group.append(de_idx)
            groups[g_idx] = new_group
      return dictionary, modify_these
    else:
      raise KeyError('Unrecognized reset type')
  elif coding_mode == 'convolutional':
    raise NotImplementedError('Not yet implemented for convolutional dict')
  else:
    raise KeyError('Unrecognized coding mode')
