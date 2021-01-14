"""
Subspace Iterative Shrinkage/Thresholding for fc sparse inference

This inference algorithm applies a thresholding operation to the *norm* of a
group of coefficients. In the case where each group is of size 1, it reduces
to vanilla ISTA/FISTA. This implementation is currently about an order of
magnitude slower than vanilla FISTA, it's likely possible to make it a lot
faster

For more context, see the following references:
.. [1] Yuan, M. & Lin, Y. (2006) Model selection and estimation in regression
       with grouped variables. Journal of the Royal Statistical Society:
       Series B (Statistical Methodology), 68(1), 49-67.
.. [2] Charles, A.S., Garrigues, P., & Rozell, C.J. (2011) Analog sparse
       approximation with applications to compressed sensing. arXiv preprint
       arXiv:1111.4118.
.. [3] Beck, A., & Teboulle, M. (2009). A fast iterative
       shrinkage-thresholding algorithm for linear inverse problems.
       SIAM Journal on Imaging Sciences, 2(1), 183–202.
"""
import torch

def run(images, dictionary, group_assignments, sparsity_weight,
        num_iters, variant='fista', ret_summed_gduplicates=True,
        initial_codes=None, early_stopping_epsilon=None, hard_threshold=False):
  """
  Runs steps of subspace Iterative Shrinkage/Thresholding.

  Compare to ./ista_fista.py. Here, thresholding is applied to the norm of a
  group of coefficients. This is also called Group LCA [2], and is an algorithm
  proposed for solving the so-called Group LASSO [1].

  Parameters
  ----------
  images : torch.Tensor(float32, size=(b, n))
      An batch of images (probably just small patches) that we want to find the
      sparse code for. n is the size of each image and b is the number of imgs.
  dictionary : torch.Tensor(float32, size=(s, n))
      This is the dictionary of basis functions that we can use to describe the
      images. n is the size of each image and s in the size of the code.
  group_assignments : list(array_like)
      Elements of this list identify which dictionary elements belong to
      a particular group. Our convention is the following: Suppose we have
        group_assignments = [[0, 2, 5], [1], [2, 3, 4, 5]]
      This specifies three groups. group 0 is comprised of elements 0, 2,
      and 5 from the dictioanary, group 1 is composed of element 1, and
      group 2 is composed of elements 2, 3, 4, and 5. Notice that each group
      can be of a different size and elements of the dictionary can
      participate in multiple groups.
  sparsity_weight : torch.Tensor(float32)
      This is the weight on the sparsity cost term in the sparse coding cost
      function. It is often denoted as \lambda
  num_iters : int
      Number of steps of ISTA to run.
  variant : str, optional
      One of {'ista', 'fista'}. Fista is the "accelerated" version of ista.
      Default 'fista'.
  ret_summed_gduplicates : bool, optional
      Suppose certain dictionary elements participate in multiple groups.
      Because synthesis is linear, each of the duplicated code values can
      be added together. And then torch.mm(codes, dictionary) gives us
      reconstructions. This is the default, especially for training. However,
      if False, then we leave the code duplicates around (but we also return
      a corresponding dictionary, which will be duplicated in these places
      too). Default True.
  # participate in multiple groups have these values simply added together
  initial_codes : torch.Tensor(float32, size=(b, s)), optional
      Start with these initial values when computing the codes. Default None.
  early_stopping_epsilon : float, optional
      Terminate if code changes by less than this amount per component,
      normalized by stepsize. Beware, requires some overhead computation.
      Default None.
  hard_threshold : bool, optional
      The hard thresholding function is the identity outside of the zeroed
      region. Default False.

  Returns
  -------
  codes : torch.Tensor(float32, size=(b, s))
      The set of codes for this set of images. s is the code size and b in the
      batch size.
  """
  # TODO: Make a simplified version for when we know there's
  #       no multiple-participation
  assert variant in ['ista', 'fista']
  # The difference between ISTA and FISTA is *where* we calculate the gradient
  # and make a steepest-descent update. In ISTA this is just the previous
  # estimate for the codes. In FISTA, we keep a set of auxilliary points which
  # combine the past two updates. See [1] for a nice description.

  # in order to facilitate parallel updates to each group, thereby minimizing
  # copies, we form an order-3 tensor which rearranges the codes into their
  # groups. This requires zero-padding for smaller groups, but is well worth it
  max_group_size = max([len(x) for x in group_assignments])
  grouped_grad_eval_pt_tensor = images.new_zeros(
      images.size(0), len(group_assignments), max_group_size)
  if initial_codes is not None:  # warm restart, we'll begin with these values
    for g_idx in range(len(group_assignments)):
      grouped_grad_eval_pt_tensor[:, g_idx, :len(group_assignments[g_idx])] = (
          initial_codes[:, group_assignments[g_idx]])

  # the grouped_dictionary tensor allows us to quickly compute a matrix
  # multiply with the grouped codes; it will repeat a dictionary element for
  # any that participate in multiple groups and also has zeros corresponding
  # to smaller subgroups.
  grouped_dictionary = images.new_zeros(
      max_group_size * len(group_assignments), dictionary.size(1))
  for g_idx in range(len(group_assignments)):
    l_idx = g_idx * max_group_size
    h_idx = l_idx + len(group_assignments[g_idx])
    grouped_dictionary[l_idx:h_idx] = dictionary[group_assignments[g_idx]]
  # Now compute a upper bound on the Hessian, as we do in ISTA/FISTA, but using
  # this new grouped dictionary. TODO: verify this is the correct thing to
  # do for the subspace variant.
  try:
    lipschitz_constant = torch.symeig(
        torch.mm(grouped_dictionary.t(), grouped_dictionary))[0][-1]
  except RuntimeError:
    print('symeig threw an exception. Likely due to one of the dictionary',
          'elements overflowing. The norm of each dictionary element is')
    print(torch.norm(dictionary, dim=1, p=2))
    raise RuntimeError()
  stepsize = 1. / lipschitz_constant

  if early_stopping_epsilon is not None:
    avg_per_component_delta = float('inf')
    if variant == 'ista':
      old_grouped_codes_tensor = torch.zeros_like(
          grouped_grad_eval_pt_tensor).copy_(grouped_grad_eval_pt_tensor)
  if variant == 'fista':
    old_grouped_codes_tensor = torch.zeros_like(
        grouped_grad_eval_pt_tensor).copy_(grouped_grad_eval_pt_tensor)
    t_kplusone = 1.
  stop_early = False
  iter_idx = 0
  while (iter_idx < num_iters and not stop_early):
    if variant == 'fista':
      t_k = t_kplusone

    ###### Subspace proximal update #######
    # gradient of l2 term is <(<gep, dictionary> - images), dictionary.T>.
    # here we use the grouped version of codes and dictionary but the math
    # is the same. It's possible this could be made faster.
    grouped_codes_tensor = grouped_grad_eval_pt_tensor - stepsize * (
      torch.mm(torch.mm(grouped_grad_eval_pt_tensor.view(
          grouped_grad_eval_pt_tensor.size(0), -1),
        grouped_dictionary) - images, grouped_dictionary.t()).view(
          grouped_grad_eval_pt_tensor.size()))
    group_norms = torch.norm(grouped_codes_tensor, p=2, dim=2, keepdim=True)
    group_norms[group_norms == 0] = 1.0  # avoid divide by zero
    # now theshold the group norms. See [1] and [2].
    if hard_threshold:
      raise NotImplementedError('TODO')
    else:
      grouped_codes_tensor.mul_(
          torch.clamp(1 - (sparsity_weight * stepsize / group_norms), min=0.))

    if variant == 'fista':
      t_kplusone = (1 + (1 + (4 * t_k**2))**0.5) / 2
      beta_kplusone = (t_k - 1) / t_kplusone
      change_in_codes = grouped_codes_tensor - old_grouped_codes_tensor
      grouped_grad_eval_pt_tensor = (grouped_codes_tensor +
                                     beta_kplusone*(change_in_codes))
      #^ the above two lines are responsible for a ~30% longer per-iteration
      #  cost for FISTA as opposed to ISTA. For certain problems though, FISTA
      #  may require many fewer steps to get a solution of similar quality.
      old_grouped_codes_tensor.copy_(grouped_codes_tensor)
    else:
      grouped_grad_eval_pt_tensor = grouped_codes_tensor

    if early_stopping_epsilon is not None:
      if variant == 'fista':
        avg_per_component_delta = torch.mean(
            torch.abs(change_in_codes) / stepsize)
      else:
        avg_per_component_delta = torch.mean(torch.abs(
          grouped_codes_tensor - old_grouped_codes_tensor) / stepsize)
        old_grouped_codes_tensor.copy_(grouped_codes_tensor)
      stop_early = (avg_per_component_delta < early_stopping_epsilon
                    and iter_idx > 0)

    iter_idx += 1

  if ret_summed_gduplicates:
    codes = images.new_zeros(images.size(0), dictionary.size(0))
    for g_idx in range(len(group_assignments)):
      codes[:, group_assignments[g_idx]] = (
          codes[:, group_assignments[g_idx]] +
          grouped_codes_tensor[:, g_idx, :len(group_assignments[g_idx])])
    return codes
  else:
    raise NotImplementedError('TODO')
