"""
Updates dictionary with steepest descent for convolutional sparse coding.

What I mean by convolutional is that the basis functions are convolved with
the sparse codes to produce an image. The basis functions will be much smaller
than the images.
"""
import torch

from utils.convolutions import create_mask

def run(images_padded, dictionary, codes, kernel_stride, padding_dims,
        stepsize=0.001, num_iters=1, normalize_dictionary=True):
  """
  Runs num_iters steps of SC steepest descent on the dictionary elements

  The a comment on axis ordering and the padding necessary to deal with
  boundary affects, see the docstrings in analysis_transforms/convolutional

  Parameters
  ----------
  images_padded : torch.Tensor(float32, size=(b, c, h, w))
      A batch of images that we want to find the CONVOLUTIONAL sparse code
      for. b is the number of images. c is the number of image channels, h is
      the (padded) height of the image, while w is the (padded) width.
  dictionary : torch.Tensor(float32, size=(s, c, kh, kw))
      The dictionary of basis functions which we can use to describe the
      images. s is the number of basis functions, the number of channels in the
      resultant code. c is the number of image channels and consequently the
      number of channels for each basis function. kh is the kernel height in
      pixels, while kw is the kernel width.
  codes : torch.Tensor(float32, size=(b, s, sh, sw))
      The inferred convolutional codes for this set of images. b is the number
      of images. s is the number of basis functions, the number of channels in
      the resultant code. sh is the height of the code. sw is the width of the
      code. These can both be inferred from the image size and kernel size.
  kernel_stride : tuple(int, int)
      The stride of the kernels in the vertical direction is kernel_stride[0]
      whereas stride in the horizontal direction is kernel_stride[1]
  padding_dims : tuple(tuple(int, int), tuple(int, int))
      The amount of padding that was done to the images -- is used to determine
      the mask. padding_dims[0] is vertical padding and padding_dims[1] is
      horizontal padding. The first component of each of these is the leading
      padding while the second component is the trailing padding.
  stepsize : torch.Tensor(float32), optional
      The step size for each iteration of steepest descent. Keep this small.
      Default 0.001.
  num_iters : int, optional
      Number of steps of steepest descent to run. Default 1
  normalize_dictionary : bool, optional
      If true, we normalize each dictionary element to have l2 norm equal to 1
      before we return. Default True.
  """
  reconstruction_mask = create_mask(images_padded, padding_dims)
  codes_temp_transposed = codes.transpose(dim0=1, dim1=0)
  # TODO: Figure out if I can remove the double-transpose in gradient comp
  for iter_idx in range(num_iters):
    # WARNING: this gradient computation can overflow, adjusting the stepsize
    # or the scale of the data are typical remedies
    gradient = (torch.nn.functional.conv2d(
      (reconstruction_mask * (
        torch.nn.functional.conv_transpose2d(codes, dictionary,
          stride=kernel_stride) - images_padded)).transpose(dim0=1, dim1=0),
      codes_temp_transposed, dilation=kernel_stride) /
      images_padded.shape[0]).transpose(dim0=1, dim1=0)
    # it makes sense to put this update on the same scale as the dictionary
    # so that stepsize is effectively dimensionless
    gradient.mul_(dictionary.norm(p=2) / gradient.norm(p=2))
    dictionary.sub_(stepsize * gradient)
    if normalize_dictionary:
      dictionary.div_(torch.squeeze(dictionary.norm(
        p=2, dim=(1, 2, 3)))[:, None, None, None])
