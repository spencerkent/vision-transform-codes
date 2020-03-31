"""
Updates dictionary with steepest descent for fully-connected sparse coding.

What I mean by fully-connected is that the basis functions have the same
dimensionality as the images.
"""
import torch

def run(images, dictionary, codes, stepsize=0.001, num_iters=1,
        normalize_dictionary=True):
  """
  Runs num_iters steps of SC steepest descent on the dictionary elements

  Parameters
  ----------
  images : torch.Tensor(float32, size=(b, n))
      An array of images (probably just small patches) that we want to find the
      sparse code for. n is the size of each image and b is the number of
      images in this batch
  dictionary : torch.Tensor(float32, size=(s, n))
      This is the dictionary of basis functions that we can use to descibe the
      images. n is the size of each image and s in the size of the code.
  codes : torch.Tensor(float32, size=(b, s))
      This is the current set of codes for a batch of images. s is the
      dimensionality of the code and b is the number of images in the batch
  stepsize : torch.Tensor(float32), optional
      The step size for each iteration of steepest descent. Keep this small.
      Default 0.001.
  num_iters : int, optional
      Number of steps of steepest descent to run. Default 1.
  normalize_dictionary : bool, optional
      If true, we normalize each dictionary element to have l2 norm equal to 1
      before we return. Default True.
  """
  # TODO: consider normalizing the gradient to be on the same scale as the
  #       dictionary. This makes the stepsize dimensionless
  for iter_idx in range(num_iters):
    dictionary.sub_(stepsize * (torch.mm(
      codes.t(), torch.mm(codes, dictionary) - images) / codes.size(0)))
    if normalize_dictionary:
      dictionary.div_(dictionary.norm(p=2, dim=1)[:, None])
