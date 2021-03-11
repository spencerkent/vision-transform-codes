"""
Subspace Iterative Shrinkage/Thresholding for fc sparse inference

This inference algorithm applies a thresholding operation to the *norm* of a
group of coefficients. In the case where each group is of size 1, it reduces
to vanilla ISTA/FISTA. This implementation is currently about 60% slower than
vanilla FISTA, due to overhead of computing group norms.

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

def run(images, dictionary, connectivity_matrix, sparsity_weight,
        num_iters, variant='fista', initial_codes=None,
        early_stopping_epsilon=None, hard_threshold=False):
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
  connectivity_matrix : torch.Tensor(float32, size(s, g))
      This matrix maps g groups to the code elements they are comprised of.
      A 1 exists in the (i, j) element of the matrix if the i^th code element
      participates in group j.
  sparsity_weight : torch.Tensor(float32)
      This is the weight on the sparsity cost term in the sparse coding cost
      function. It is often denoted as \lambda
  num_iters : int
      Number of steps of ISTA to run.
  variant : str, optional
      One of {'ista', 'fista'}. Fista is the "accelerated" version of ista.
      Default 'fista'.
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
  assert variant in ['ista', 'fista']
  # We can take the stepsize from the largest eigenvalue of the Gram matrix,
  # dictionary @ dictionary.T. We only need to compute this once, here. This
  # guarantees convergence but may be a bit conservative and also could be
  # expensive to compute. The matrix is of size (s, s), and s >= n,
  # so we can use the smaller covariance matrix of size (n, n), which will
  # have the same eigenvalues. One could instead perform a linesearch to
  # find the stepsize, but in my experience this does not work well.
  try:
    lipschitz_constant = torch.symeig(
        torch.mm(dictionary.t(), dictionary))[0][-1]
  except RuntimeError:
    print('symeig threw an exception. Likely due to one of the dictionary',
          'elements overflowing. The norm of each dictionary element is')
    print(torch.norm(dictionary, dim=1, p=2))
    raise RuntimeError()
  stepsize = 1. / lipschitz_constant

  # The difference between ISTA and FISTA is *where* we calculate the gradient
  # and make a steepest-descent update. In ISTA this is just the previous
  # estimate for the codes. In FISTA, we keep a set of auxilliary points which
  # combine the past two updates. See [1] for a nice description.
  if initial_codes is None:
    grad_eval_points = images.new_zeros(images.size(0), dictionary.size(0))
  else:  # warm restart, we'll begin with these values
    grad_eval_points = initial_codes

  if early_stopping_epsilon is not None:
    avg_per_component_delta = float('inf')
    if variant == 'ista':
      old_codes = torch.zeros_like(
          grad_eval_points).copy_(grad_eval_points)
  if variant == 'fista':
    old_codes = torch.zeros_like(
        grad_eval_points).copy_(grad_eval_points)
    t_kplusone = 1.
  stop_early = False
  iter_idx = 0
  while (iter_idx < num_iters and not stop_early):
    if variant == 'fista':
      t_k = t_kplusone

    ###### Subspace proximal update #######
    codes = grad_eval_points - stepsize * torch.mm(
        torch.mm(grad_eval_points, dictionary) - images, dictionary.t())
    # compute group norms
    group_norms = torch.sqrt(torch.mm(torch.pow(codes, 2), connectivity_matrix))
    group_norms[group_norms == 0] = 1.0
    # now theshold the group norms. See [1] and [2].
    if hard_threshold:
      raise NotImplementedError('TODO')
    else:
      # theshold the codes according to these group norms
      multiplier = torch.clamp(
          1 - sparsity_weight * stepsize / group_norms, min=0.)
      # connectivity_matrix.t() 'distributes' these to each element
      codes.mul_(torch.mm(multiplier, connectivity_matrix.t()))

    if variant == 'fista':
      t_kplusone = (1 + (1 + (4 * t_k**2))**0.5) / 2
      beta_kplusone = (t_k - 1) / t_kplusone
      change_in_codes = codes - old_codes
      grad_eval_points = codes + beta_kplusone*(change_in_codes)
      #^ the above two lines are responsible for a ~30% longer per-iteration
      #  cost for FISTA as opposed to ISTA. For certain problems though, FISTA
      #  may require many fewer steps to get a solution of similar quality.
      old_codes.copy_(codes)
    else:
      grad_eval_points = codes

    if early_stopping_epsilon is not None:
      if variant == 'fista':
        avg_per_component_delta = torch.mean(
            torch.abs(change_in_codes) / stepsize)
      else:
        avg_per_component_delta = torch.mean(
            torch.abs(codes - old_codes) / stepsize)
        old_codes.copy_(codes)
      stop_early = (avg_per_component_delta < early_stopping_epsilon
                    and iter_idx > 0)

    iter_idx += 1

  return codes
