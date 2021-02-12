"""
Iterative Shrinkage/Thresholding for Laplacian Scale Mixture priors

[1] Pierre paper
[2] Candes paper
[3] Adam Charles paper
[4] Zou paper

According to [2], $\beta$ should be set to slightly smaller than the expected
magnitudes of the nonzero coefficients.

According to [2] (which uses a slightly different approach than ISTA/FISTA), 
a rule of thumb for \beta is 0.1*\sigma where sigma is the stand deviation of
nonzero coefficients

Another heuristic that could be useful is 0.1 * the largest inner product 
between the dictionary and the data...this will give you 10% of the magnitude
of the coefficients after the first gradient update.
"""

# Need
# gamma, beta
# option to do interleaved updating of weight
# global weight

# ista stepsize

import torch

def run(images, dictionary, sparsity_weight, num_iters,
        update_scalvars_every=1, variant='fista',
        alpha_param=0., beta_param=1e-3, initial_codes=None,
        early_stopping_epsilon=None, nonnegative_only=False,
        hard_threshold=False):

  assert variant in ['ista', 'fista']
  assert type(update_scalvars_every) == int
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

  # reweighted ista assigns to each coefficient a scale variable that gets
  # iteratively updated along with the ista updates
  scale_variables = images.new_ones(images.size(0), dictionary.size(0))

  if early_stopping_epsilon is not None:
    avg_per_component_delta = float('inf')
    if variant == 'ista':
      old_codes = torch.zeros_like(grad_eval_points).copy_(grad_eval_points)
  if variant == 'fista':
    old_codes = torch.zeros_like(grad_eval_points).copy_(grad_eval_points)
    t_kplusone = 1.
  stop_early = False
  iter_idx = 0

  recorded_recon_error = [float(torch.mean(torch.norm(
      torch.mm(grad_eval_points, dictionary) - images, p=2, dim=1)).cpu())]
  recorded_l1 = [float(torch.mean(torch.norm(
      scale_variables * grad_eval_points, p=1, dim=1)).cpu())]
  recorded_bpdn_loss = [recorded_recon_error[-1] +
                        sparsity_weight*recorded_l1[-1]]
  recorded_l0 = [float(torch.mean(torch.norm(
      grad_eval_points, p=0, dim=1)).cpu())]
  while (iter_idx < num_iters and not stop_early):
    if variant == 'fista':
      t_k = t_kplusone

    #### Proximal update ####
    codes = grad_eval_points - stepsize * torch.mm(
        torch.mm(grad_eval_points, dictionary) - images, dictionary.t())
    # the threshold of the is adjusted by the scale variable
    if hard_threshold:
      if nonnegative_only:
        codes[codes < (sparsity_weight*stepsize*scale_variables)] = 0
      else:
        codes[torch.abs(codes) <
              (sparsity_weight*stepsize*scale_variables)] = 0
    else:
      if nonnegative_only:
        codes.sub_(sparsity_weight * stepsize * scale_variables).clamp_(min=0.)
        #^ shifted rectified linear activation
      else:
        pre_threshold_sign = torch.sign(codes)
        codes.abs_()
        codes.sub_(sparsity_weight * stepsize * scale_variables).clamp_(min=0.)
        codes.mul_(pre_threshold_sign)
        #^ now contains the "soft thresholded" (non-rectified) output x_{k+1}

    if iter_idx != 0 and (iter_idx % update_scalvars_every == 0):
      scale_variables = (alpha_param + 1) / (beta_param + codes.abs())

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

    recorded_recon_error.append(float(torch.mean(torch.norm(
        torch.mm(codes, dictionary) - images, p=2, dim=1)).cpu()))
    recorded_l1.append(float(torch.mean(torch.norm(
        scale_variables * codes, p=1, dim=1)).cpu()))
    recorded_l0.append(float(torch.mean(torch.norm(
        codes, p=0, dim=1)).cpu()))
    recorded_bpdn_loss.append(recorded_recon_error[-1] + 
                              sparsity_weight*recorded_l1[-1])

    iter_idx += 1

  return codes, recorded_recon_error, recorded_l1, recorded_l0, recorded_bpdn_loss
