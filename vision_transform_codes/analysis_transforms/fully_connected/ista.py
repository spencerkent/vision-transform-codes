"""
Iterative Soft Thresholding for fully-connected sparse inference

What I mean by fully-connected is that the basis functions have the same
dimensionality as the images.
"""
import torch

def run(images, dictionary, sparsity_weight, num_iters,
        initial_codes=None, early_stopping_epsilon=None,
        nonnegative_only=False):
  """
  Runs steps of Iterative Soft Thresholding with a constant stepsize

  Computes ISTA updates on samples in parallel. Written to
  minimize data copies, for speed. Ideally, one could stop computing updates
  on a sample whose code is no longer changing very much. However, the
  copying overhead this requires (to only update a newer, smaller set of
  samples) seems to overpower the savings. Further optimization may be
  possible.

  Parameters
  ----------
  images : torch.Tensor(float32, size=(n, b))
      An batch of images (probably just small patches) that we want to find the
      sparse code for. n is the size of each image and b is the number of imgs.
  dictionary : torch.Tensor(float32, size=(n, s))
      This is the dictionary of basis functions that we can use to describe the
      images. n is the size of each image and s in the size of the code.
  sparsity_weight : torch.Tensor(float32)
      This is the weight on the sparsity cost term in the sparse coding cost
      function. It is often denoted as \lambda
  num_iters : int
      Number of steps of ISTA to run.
  initial_codes : torch.Tensor(float32, size=(s, b)), optional
      Start with these initial values when computing the codes. Default None.
  early_stopping_epsilon : float, optional
      Terminate if code changes by less than this amount per component,
      normalized by stepsize. Beware, requires some overhead computation.
      Default None.
  nonnegative_only : bool, optional
      If true, our code values can only be nonnegative. We just chop off the
      left half of the ISTA soft thresholding function and it becomes a
      shifted RELU function. The amount of the shift from a generic RELU is
      the sparsity_weight times the (inferred) stepsize. Default False.

  Returns
  -------
  codes : torch.Tensor(float32, size=(s, b))
      The set of codes for this set of images. s is the code size and b in the
      batch size.
  """
  # We can take the stepsize from the largest eigenvalue of the Gram matrix,
  # dictionary.T @ dictionary. We only need to compute this once, here. This
  # guarantees convergence but may be a bit conservative and also could be
  # expensive to compute. The matrix is of size (s, s), and s >= n,
  # so we can use the smaller covariance matrix of size (n, n), which will
  # have the same eigenvalues. One could instead perform a linesearch to
  # find the stepsize, but in my experience this does not work well.
  try:
    lipschitz_constant = torch.symeig(
        torch.mm(dictionary, dictionary.t()))[0][-1]
  except:
    print('symeig threw an exception. Likely due to one of the dictionary',
          'elements overflowing. The norm of each dictionary element is')
    print(torch.norm(dictionary, dim=0, p=2))
  stepsize = 1. / lipschitz_constant

  if initial_codes is None:
    codes = images.new_zeros(dictionary.size(1), images.size(1))
  else:
    codes = initial_codes  # warm restart, we'll begin with these values

  if early_stopping_epsilon is not None:
    old_codes = torch.zeros_like(codes).copy_(codes)
    avg_per_component_delta = float('inf')
  stop_early = False
  iter_idx = 0
  while (iter_idx < num_iters and not stop_early):

    ###### Proximal step #######
    # gradient of l2 term is <dictionary^T, (<dictionary, codes> - images)>
    codes.sub_(stepsize * torch.mm(dictionary.t(),
                                   torch.mm(dictionary, codes) - images))
    if nonnegative_only:
      codes.sub_(sparsity_weight * stepsize).clamp_(min=0.)
      #^ shifted rectified linear activation
    else:
      pre_threshold_sign = torch.sign(codes)
      codes.abs_()
      codes.sub_(sparsity_weight * stepsize).clamp_(min=0.)
      codes.mul_(pre_threshold_sign)
      #^ now contains the "soft thresholded" (non-rectified) output
    ###### Proximal step #######

    if early_stopping_epsilon is not None:
      avg_per_component_delta = torch.mean(
          torch.abs(codes - old_codes) / stepsize)
      stop_early = (avg_per_component_delta < early_stopping_epsilon
                    and iter_idx > 0)
      old_codes.copy_(codes)

    iter_idx += 1

  return codes
