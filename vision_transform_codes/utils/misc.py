"""
Miscellaneous utils that might be useful
"""
import os
import pickle
import numpy as np

def load_newest_dictionary_checkpoint(checkpoint_dir):
  # get the dictionary from the highest checkpoint iteration
  # this is fragile and could be improved significantly
  iter_nums = []
  for _, _, filenames in os.walk(checkpoint_dir):
    for filename in filenames:
      if filename[:27] == 'checkpoint_dictionary_iter_':
        iter_nums.append(int(filename[27:]))
    break
  print('checkpoint idx: ', max(iter_nums))
  return pickle.load(open(
    checkpoint_dir / ('checkpoint_dictionary_iter_' + str(max(iter_nums))),
    'rb'))

def rotational_average(array_2d, nbins=10, elem_cartesian_coords=None):
  """
  Take the 'rotational average' of a 2d array

  Treats an array as a cartesian sampling of 2d space, converts to polar
  coordinates and then averages the values across angle

  Parameters
  ----------
  array_2d : ndarray
      The array to be rotationally averaged
  nbins : int, optional
      We discretize radius into this many bins. Default 10
  elem_cartesian_coords : tuple(ndarray, ndarray)
      The cartesian coordinates to associate with each element in
      array_2d. elem_cartesian_coords[0] are the vertical values and
      elem_cartesian_coords[1] are the horizontal values, like we might
      get from np.meshgrid. Otherwise we treat each element as having
      cartesian coords given by its position in the array. Default None.

  Returns
  -------
  rotational_means : ndarray(size=(nbins,))
      Means across polar angle for the nbins different polar magnitudes
  magnitude_bins : ndarray(size=(nbins,))
      The left edge of each bin.
  """
  def cartesian_to_polar(x, y):
    mag = np.sqrt(x**2 + y**2)
    phase = np.arctan2(y, x)
    return (mag, phase)

  if elem_cartesian_coords is None:
    v_coords, h_coords = np.meshgrid(np.arange(array_2d.shape[0]),
                                     np.arange(array_2d.shape[1]),
                                     indexing='ij')
  else:
    v_coords, h_coords = elem_cartesian_coords

  polar_mag, polar_phase = cartesian_to_polar(h_coords, v_coords)
  highest_valid_mag = max((np.max(np.abs(h_coords)), np.max(np.abs(v_coords))))
  # ^don't consider magnitudes larger than the largest cartesian
  #  dimension -- these are in the 'corners'
  discrete_polar_mag = np.linspace(0.0, highest_valid_mag, nbins + 1)
  bin_assignments = np.digitize(polar_mag, discrete_polar_mag) - 1
  bin_assignments[polar_mag == highest_valid_mag] = nbins - 1
  # ^include the right edge of the highest bin
  rotational_means = np.zeros(nbins)
  for mag_idx in range(nbins):
    rotational_means[mag_idx] = np.mean(array_2d[bin_assignments == mag_idx])
  # can make this much faster by sorting bin_assignments first

  return rotational_means, discrete_polar_mag[:-1]
