"""
Filter matrix is precisely the matrix inverse of the dictionary matrix
"""
import torch

def run(images, dictionary):
  """
  Infers the code using the exact matrix inverse of the dictionary matrix

  Parameters
  ----------
  images : torch.Tensor(float32, size=(n, b))
      An array of images (probably just small patches) to find the direct
      linear code for. n is the size of each image and b is the number
      of images in this batch
  dictionary : torch.Tensor(float32, size=(n, n))
      This is the dictionary of basis functions that we can use to descibe the
      images. n is the size of each image and is also the size of the code.
  """
  return torch.mm(torch.inverse(dictionary), images)
