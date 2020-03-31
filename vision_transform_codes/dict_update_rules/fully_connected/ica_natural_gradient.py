"""
This updates the dictionary according to the ICA natural gradient
"""
import torch

def run(dictionary, codes, stepsize=0.001, num_iters=1):
  """
  Updates the dictionary according to the ICA natural gradient learning rule

  Notice that we do not need any images in order to calculate the natural
  gradient update.

  Parameters
  ----------
  dictionary : torch.Tensor(float32, size=(s, n))
      This is the dictionary of basis functions that we can use to descibe the
      images. n is the size of each image and s in the size of the code.
  codes : torch.Tensor(float32, size=(b, s))
      This is the current set of codes for a batch of images. s is the
      dimensionality of the code and b is the number of images in the batch
  stepsize : torch.Tensor(float32)
      The step size for each iteration of natural gradient. Keep this small
  num_iters : int
      Number of steps of natural gradient update to run
  """
  eye_mat = torch.eye(codes.size(1), device=codes.device)
  for iter_idx in range(num_iters):
    # the update for a single patch is <(<s.T, z> - I), A> where s is the code,
    # z is the sign of the code, I is the identity matrix, and A is the
    # dictionary. We can average this update over a batch of images which is
    # what we do below:
    dict_update = stepsize * (
        torch.mm((torch.mm(codes.t(), torch.sign(codes)) / codes.size(0))
                  - eye_mat, dictionary))
    dictionary.add_(dict_update)  # we want to *ascend* the gradient
