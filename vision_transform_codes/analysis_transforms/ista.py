"""
Implementations of Iterative Soft Thresholding
"""
import torch

def run(images, dictionary, sparsity_weight, num_iters, nonnegative_only=False):
  """
  Runs num_iters steps of Iterative Soft Thresholding

  Parameters
  ----------
  images : torch.Tensor(float32, size=(n, b))
      An array of images (probably just small patches) that to find the sparse
      code for. n is the size of each image and b is the number of images in
      this batch
  dictionary : torch.Tensor(float32, size=(n, s))
      This is the dictionary of basis functions that we can use to descibe the
      images. n is the size of each image and s in the size of the code.
  sparsity_weight : torch.Tensor(float32)
      This is the weight on the sparsity cost term in the sparse coding cost
      function. It is often denoted as \lambda
  num_iters : int
      Number of steps of ISTA to run
  nonnegative_only : bool, optional
      If true, our code values can only be nonnegative. We just chop off the
      left half of the ISTA soft thresholding function and it becomes a
      shifted RELU function. The amount of the shift from a generic RELU is
      precisely the sparsity_weight. Default False

  Returns
  -------
  codes : torch.Tensor(float32, size=(s, b))
      The set of codes for this set of images. s is the code size and b in the
      batch size.
  """
  # Stepsize set by the largest eigenvalue of the Gram matrix. Since this is
  # of size (s, s), and s >= n, we want to use the covariance matrix
  # because it will be of size (n, n) and have the same eigenvalues
  lipschitz_constant = torch.symeig(
      torch.mm(dictionary, dictionary.t()))[0][-1]
  stepsize = 1. / lipschitz_constant

  codes = images.new_zeros(dictionary.size(1), images.size(1))
  for iter_idx in range(num_iters):
    # gradient of l2 term is <dictionary^T, (<dictionary, codes> - images)>
    codes.sub_(stepsize * torch.mm(dictionary.t(),
                                   torch.mm(dictionary, codes) - images))
    #^ pre-threshold values s - lambda*A^T(As - x)
    if nonnegative_only:
      codes.sub_(sparsity_weight * stepsize).clamp_(min=0.)
      #^ shifted rectified linear activation
    else:
      pre_threshold_sign = torch.sign(codes)
      codes.abs_()
      codes.sub_(sparsity_weight * stepsize).clamp_(min=0.)
      codes.mul_(pre_threshold_sign)
      #^ now contains the "soft thresholded" (non-rectified) output
  return codes
