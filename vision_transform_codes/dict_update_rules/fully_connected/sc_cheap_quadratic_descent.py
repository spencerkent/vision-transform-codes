"""
Updates dictionary with a modified descent for fully-connected sparse coding.

What I mean by fully-connected is that the basis functions have the same
dimensionality as the images. This uses the diagonal of the Hessian to
rescale the updates. It's like getting a quadratic update for each coefficient
independently, while not computing the full hessian.
"""
import torch

def run(images, dictionary, codes, hessian_diagonal, stepsize=0.001,
        num_iters=1, lowest_code_val=0.001, normalize_dictionary=True):
  """
  Runs num_iters steps of an approximate quadratic descent

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
  hessian_diagonal : torch.Tensor(float32, size=(s,))
      An estimate of the diagonal of the hessian that we'll compute outside of
      this function call.
  stepsize : torch.Tensor(float32)
      The step size for each iteration of the quad. descent. Keep this small.
      Default 0.001.
  num_iters : int, optional
      Number of steps of quad. descent to run. Default 1.
  lowest_code_val : float, optional
      Used to condition the hessian diagonal to not be too small. Default 0.001
  normalize_dictionary : bool, optional
      If true, we normalize each dictionary element to have l2 norm equal to 1
      before we return. Default True.
  """
  for iter_idx in range(num_iters):
    dict_update = stepsize * (torch.mm(
      codes.t(), torch.mm(codes, dictionary) - images) / codes.size(0))
    dict_update.div_(hessian_diagonal[:, None] + lowest_code_val)
    dictionary.sub_(dict_update)
    if normalize_dictionary:
      dictionary.div_(dictionary.norm(p=2, dim=1)[:, None])
