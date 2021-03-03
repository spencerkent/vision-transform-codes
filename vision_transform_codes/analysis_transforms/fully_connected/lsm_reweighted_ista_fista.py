"""
Iterative Shrinkage/Thresholding for SC with Laplacian Scale Mixture priors

This variant of sparse coding encodes 'second-layer' variables that
set the scale of the laplacian prior on each coefficient. In this way they
can be thought of as "hyperpriors." The prior on each coefficient is actually
a 'scale-mixture' of laplacians. We can impose structure on the hyperpriors,
thereby inducing structure on the coefficients they connect to. The most
direct explication of these ideas can be found in [1], with further context
in [2]. In the statistics literature, namely [3] and [4], the inference
algorithm we use in the 'factored' hyperprior case was proposed to promote
sparsity, but without the Bayesian formalism of later works.

.. [1] Garrigues, P. & Olshausen, B. (2010). Group Sparse Coding with a
       Laplacian Scale Mixture Prior. NIPS 2010.
.. [2] Charles, A., Garrigues, P., & Rozell, C. (2011). Analog sparse
       approximation with applications to compressed sensing. ArXiv preprint
       arXiv:1111.4118v1.
.. [3] Zou, H. (2006). The adaptive lasso and its oracle properties. Journal
       or the American Statistical Association, 101(476).
.. [4] Candes, E., Wakin, M., & Boyd, S. (2008). Enhancing sparsity by
       reweighted l1 minimization. Journal of Fourier Analysis and Applications.

"""
# Just a couple misc. notes for further followup:
# According to [4] (which uses a slightly different approach than ISTA/FISTA),
# a rule of thumb for \beta is 0.1*\sigma where sigma is the stand deviation of
# nonzero coefficients. Another heuristic that could be useful is 0.1 * the
# largest inner product between the dictionary and the data...this will give
# you 10% of the magnitude of the coefficients after the first gradient update.
import torch
from utils import topographic

def run(images, dictionary, sparsity_weight, num_iters,
        update_rate_vars_every=10, variant='fista',
        alpha_param=0., beta_param=1e-3, initial_codes=None,
        early_stopping_epsilon=None, nonnegative_only=False,
        hard_threshold=False, rate_var_structure='factored',
        structure_params=None, return_rate_variables=False):
  """
  Runs steps of iteratively reweighted ISTA/FISTA for LSM sparse coding

  Parameters
  ----------
  images : torch.Tensor(float32, size=(b, n))
      An batch of images (probably just small patches) that we want to find the
      sparse code for. n is the size of each image and b is the number of imgs.
  dictionary : torch.Tensor(float32, size=(s, n))
      This is the dictionary of basis functions that we can use to describe the
      images. n is the size of each image and s in the size of the code.
  sparsity_weight : torch.Tensor(float32)
      This is the weight on the sparsity cost term in the sparse coding cost
      function. It is often denoted as \lambda
  num_iters : int
      Number of steps of ISTA/FISTA to run.
  update_rate_vars_every : int, optional
      We run some number of ISTA steps before updating the rate variable,
      then repeat. Doing this every iteration is fairly aggressive, but can
      work. Default 10.
  variant : str, optional
      One of {'ista', 'fista'}. Fista is the "accelerated" version of ista.
      Default 'fista'.
  alpha_param : float, optional
      Parameterizes a gamma distribution on the hyperprior. Setting this to
      0 and keeping beta small, generates approximately a truncated inverse
      power-law distribution. Default 0.
  beta_param : float, optional
      The beta parameter of the gamma hyperprior. See note above. Default 0.001
  initial_codes : torch.Tensor(float32, size=(b, s)), optional
      Start with these initial values when computing the codes. Default None.
  early_stopping_epsilon : float, optional
      Terminate if code changes by less than this amount per component,
      normalized by stepsize. Beware, requires some overhead computation.
      Default None.
  nonnegative_only : bool, optional
      If true, our code values can only be nonnegative. We just chop off the
      left half of the ISTA thresholding function and it becomes a
      shifted RELU function. The amount of the shift from a generic RELU is
      the sparsity_weight times the (inferred) stepsize. Default False.
  hard_threshold : bool, optional
      The hard thresholding function is the identity outside of the zeroed
      region. Default False.
  rate_var_structure : str, optional
      One of ['factored', 'disjoint_groups', 'topographic']. With factored
      structure, no rate variables are shared between coefficients. With the
      disjoint groups variant, there are a group of coefficients that all
      have the same rate variable. The topographic variant imposes a
      discretely-sampled taurus (which we're calling our topography) on the
      rate variables, with potentially some amount of overlap between
      coefficients. This means the coefficients have a topography too. See
      utils/topographic.py and utils/plotting.py for more context. Default None
  structure_params : dictionary, optional
      When the rate variable structure is either 'disjoint_groups' or
      'topographic', we need the following:
        'connectivity_matrix' : torch.Tensor(float32, size=(s, l))
          An s x l matrix that maps coefficients to their hierarchical rate
          variables. l is the (possibly smaller) number of rate variables. It
          will just sum over the rate variable's "neighborhood." This matrix
          should be pre-generated with utils/topographic.py
        'spreading_matrix' : torch.Tensor(float32, size=(l, s))
          Just a normalized transpose of the connectivity matrix, used to
          distributed all hierarchical rate variables to the coefficients.
          See examples.
        'neighborhood_sizes' : torch.Tensor(float32, size=(1, l))
          Indicates the neighborhood size for each hierarchical rate variable.
  return_rate_variables : bool, optional
      If true, we not only return the unnormalized code values, but the rate
      variables, which when pointwise-multiplied actually normalize the these
      to unit-scale laplacians. We also return the hierarchical representation
      of the rate variables, which, unlike the non-hierarchical versions, have
      no redundancy. Default False.

  Returns
  -------
  codes :
  codes : torch.Tensor(float32, size=(b, s))
      The set of codes for this set of images. s is the code size and b in the
      batch size.
  rate_variables : torch.Tensor(float32, size=(b, s))
      These are the computed rates (AKA inverse-scales) for each coefficient.
      You can pointwise multiply codes * rate_variables to get *normalized*
      versions of the code, which have unit laplacian prior.
  hierarchical_rate_variables : torch.Tensor(float32, size=(b, l))
      These are the 'second-layer' rate variables which, project to the first
      layer. The may have much less redunancy that the pointwise rate_variables.
      You can recover rate_variables by post multiplying
      hierarchical_rate_variables with the spreading matrix.
  """
  assert variant in ['ista', 'fista']
  assert type(update_rate_vars_every) == int
  assert rate_var_structure in ['factored', 'disjoint_groups', 'topographic']
  if rate_var_structure != 'factored':
    assert structure_params is not None
    assert 'connectivity_matrix' in structure_params
    assert 'spreading_matrix' in structure_params
    assert 'neighborhood_sizes' in structure_params
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

  # reweighted ista assigns to each coefficient a rate (AKA inverse-scale)
  # variable that gets iteratively updated along with the ista updates
  rate_variables = images.new_ones(images.size(0), dictionary.size(0))

  if early_stopping_epsilon is not None:
    avg_per_component_delta = float('inf')
    if variant == 'ista':
      old_codes = torch.zeros_like(grad_eval_points).copy_(grad_eval_points)
  if variant == 'fista':
    old_codes = torch.zeros_like(grad_eval_points).copy_(grad_eval_points)
    t_kplusone = 1.
  stop_early = False
  iter_idx = 0
  while (iter_idx < num_iters and not stop_early):
    if variant == 'fista':
      t_k = t_kplusone

    #### Proximal update ####
    codes = grad_eval_points - stepsize * torch.mm(
        torch.mm(grad_eval_points, dictionary) - images, dictionary.t())
    # reweighted ISTA/FISTA changes individual thresholds periodically
    if hard_threshold:
      if nonnegative_only:
        codes[codes < (sparsity_weight*stepsize*rate_variables)] = 0
      else:
        codes[torch.abs(codes) <
              (sparsity_weight*stepsize*rate_variables)] = 0
    else:
      if nonnegative_only:
        codes.sub_(sparsity_weight * stepsize * rate_variables).clamp_(min=0.)
        #^ shifted rectified linear activation
      else:
        pre_threshold_sign = torch.sign(codes)
        codes.abs_()
        codes.sub_(sparsity_weight * stepsize * rate_variables).clamp_(min=0.)
        codes.mul_(pre_threshold_sign)
        #^ now contains the "soft thresholded" (non-rectified) output x_{k+1}

    # The next chunk is what makes this different from vanilla ISTA/FISTA:
    if iter_idx != 0 and (iter_idx % update_rate_vars_every == 0):
      if rate_var_structure == 'factored':
        rate_variables = (alpha_param + 1) / (beta_param + codes.abs())
        # ^this is equation (9) from [1]. It corresponds to a gamma hyperprior.
      else:
        hierarchical_rate_variables = (
            alpha_param + structure_params['neighborhood_sizes']) / (
            beta_param + torch.mm(torch.abs(codes),
              structure_params['connectivity_matrix']))
        rate_variables = torch.mm(hierarchical_rate_variables,
                                  structure_params['spreading_matrix'])

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

  if return_rate_variables:
    if rate_var_structure == 'factored':
      hierarchical_rate_variables = rate_variables  # there is no hierarchy...
    return codes, rate_variables, hierarchical_rate_variables
  else:
    return codes
