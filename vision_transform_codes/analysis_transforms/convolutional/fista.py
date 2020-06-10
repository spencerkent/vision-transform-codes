"""
Fast Iterative Shrinkage/Thresholding for convolutional sparse inference

What I mean by convolutional is that the basis functions are convolved with
the sparse codes to produce an image. The basis functions will be much smaller
than the images.
"""
import torch

from utils.convolutions import code_dim_from_padded_img_dim
from utils.convolutions import create_mask

def run(images_prepadded, dictionary, kernel_stride, padding_dims,
        sparsity_weight, num_iters, initial_codes=None,
        early_stopping_epsilon=None, nonnegative_only=False):
  """
  Runs steps of Fast Iterative Shrinkage/Thresholding with a constant stepsize

  Computes FISTA updates on samples in parallel. Written to
  minimize data copies, for speed. Ideally, one could stop computing updates
  on a sample whose code is no longer changing very much. However, the
  copying overhead this requires (to only update a newer, smaller set of
  samples) seems to overpower the savings. Further optimization may be
  possible.

  The index order convention we use is slightly different from the case of
  fully-connected sparse coding. In the case of fc sparse coding we take the
  sample axis to be the last axis because this is cleaner and more
  closely follows the way the math is usually written. In the case of
  convolutional sparse coding the sample axis is the first axis, because this
  more closely follows the typical PyTorch convention and makes the code
  cleaner. The two can be massaged to use the same convention but I think it
  is more clear to keep the fully-connected and convolutional conventions
  separate.

  To deal with boundary effects when we do convolution with overlapping
  kernels, the images are prepadded. The reconstruction error in this padded
  region is ignored via multiplication with a simple mask. Reconstructed images
  will include the padded region, but the user should strip away this when
  measuring reconstruction accuracy. This is a simple, clean, and effective
  way to deal with the boundary effects inherent in using convolutions.

  Parameters
  ----------
  images_prepadded : torch.Tensor(float32, size=(b, c, h, w))
      A batch of images that we want to find the CONVOLUTIONAL sparse code
      for. b is the number of images. c is the number of image channels, h is
      the (padded) height of the image, while w is the (padded) width.
  dictionary : torch.Tensor(float32, size=(s, c, kh, kw))
      The dictionary of basis functions which we can use to describe the
      images. s is the number of basis functions, the number of channels in the
      resultant code. c is the number of image channels and consequently the
      number of channels for each basis function. kh is the kernel height in
      pixels, while kw is the kernel width.
  kernel_stride : tuple(int, int)
      The stride of the kernels in the vertical direction is kernel_stride[0]
      whereas stride in the horizontal direction is kernel_stride[1]
  padding_dims : tuple(tuple(int, int), tuple(int, int))
      The amount of padding that was done to the images -- is used to determine
      the mask. padding_dims[0] is vertical padding and padding_dims[1] is
      horizontal padding. The first component of each of these is the leading
      padding while the second component is the trailing padding.
  sparsity_weight : float
      This is the weight on the sparsity cost term in the sparse coding cost
      function. It is often denoted as \lambda
  num_iters : int
      Number of steps of ISTA to run.
  initial_codes : torch.Tensor(float32, size=(b, s, sh, sw)), optional
      Start with these initial values when computing the codes. b is the number
      of images. s is the number of basis functions, the number of channels in
      the resultant code. sh is the height of the code. sw is the width of the
      code. These can both be inferred from the image size and kernel size.
      Default None.
  early_stopping_epsilon : float, optional
      Terminate if code changes by less than this amount per component,
      normalized by stepsize. Beware, requires some overhead computation.
      Default None.
  nonnegative_only : bool, optional
      If true, our code values can only be nonnegative. We just chop off the
      left half of the ISTA thresholding function and it becomes a
      shifted RELU function. The amount of the shift from a generic RELU is
      precisely the sparsity_weight. Default False

  Returns
  -------
  codes : torch.Tensor(float32, size=(b, s, sh, sw))
      The inferred convolutional codes for this set of images. b is the number
      of images. s is the number of basis functions, the number of channels in
      the resultant code. sh is the height of the code. sw is the width of the
      code. These can both be inferred from the image size and kernel size.
  """
  # We can take the stepsize from the largest eigenvalue of the Gram matrix,
  # which contains all the innerproducts between kernels. We only need to
  # compute this once, here. This guarantees convergence but may be a bit
  # conservative and also could be expensive to compute. If the number of
  # kernels s is > the flattened kernel size c x kh x kw, then we want to use
  # the covariance matrix instead of the gram matrix. For now we use the
  # Gram matrix. One could instead perform a linesearch to
  # find the stepsize, but in my experience this does not work well.
  flattened_dict = torch.flatten(dictionary, start_dim=1)
  gram_matrix = torch.mm(flattened_dict, flattened_dict.t())
  try:
    lipschitz_constant = torch.symeig(gram_matrix)[0][-1]
  except RuntimeError:
    print('symeig threw an exception. Likely due to one of the dictionary',
          'elements overflowing. The norm of each dictionary element is')
    print(torch.norm(dictionary, dim=[1, 2, 3], p=2))
    raise RuntimeError()
  stepsize = 1. / lipschitz_constant

  code_height = code_dim_from_padded_img_dim(
      images_prepadded.shape[2], dictionary.shape[2], kernel_stride[0])
  code_width = code_dim_from_padded_img_dim(
      images_prepadded.shape[3], dictionary.shape[3], kernel_stride[1])
  if initial_codes is None:
    codes = images_prepadded.new_zeros((
      images_prepadded.shape[0], dictionary.shape[0], code_height, code_width))
  else:
    # warm restart, we'll begin with these values
    assert initial_codes.shape[0] == images_prepadded.shape[0]
    assert initial_codes.shape[1] == dictionary.shape[0]
    assert initial_codes.shape[2] == code_height
    assert initial_codes.shape[3] == code_width
    codes = initial_codes
  reconstruction_mask = create_mask(images_prepadded, padding_dims)
  aux_points = torch.zeros_like(codes).copy_(codes)
  # ^the twist in FISTA is that we compute a proximal update using a set of
  #  auxilliary points.

  if early_stopping_epsilon is not None:
    avg_per_component_delta = float('inf')
  old_codes = torch.zeros_like(codes).copy_(codes)
  stop_early = False
  t_kplusone = 1.
  iter_idx = 0
  while (iter_idx < num_iters and not stop_early):

    t_k = t_kplusone
    ###### Proximal step #######
    # gradient of l2 term **wrt aux points** is
    # corr(dictionary, (conv(dictionary, aux) - images)), but with PyTorch's
    # weird semantics conv is conv_transpose2d and corr is conv2d...
    codes = aux_points - stepsize * torch.nn.functional.conv2d(
        reconstruction_mask * (torch.nn.functional.conv_transpose2d(
          aux_points, dictionary, stride=kernel_stride) - images_prepadded),
      dictionary, stride=kernel_stride)
    if nonnegative_only:
      codes.sub_(sparsity_weight * stepsize).clamp_(min=0.)
    else:
      pre_threshold_sign = torch.sign(codes)
      codes.abs_()
      codes.sub_(sparsity_weight * stepsize).clamp_(min=0.)
      codes.mul_(pre_threshold_sign)
    ###### Proximal step #######
    t_kplusone = (1 + (1 + (4 * t_k**2))**0.5) / 2
    beta_kplusone = (t_k - 1) / t_kplusone
    change_in_codes = codes - old_codes
    aux_points = codes + beta_kplusone*(change_in_codes)

    if early_stopping_epsilon is not None:
      avg_per_component_delta = torch.mean(
          torch.abs(codes - old_codes) / stepsize)
      stop_early = (avg_per_component_delta < early_stopping_epsilon
                    and iter_idx > 0)

    old_codes.copy_(codes)
    iter_idx += 1

  return codes
