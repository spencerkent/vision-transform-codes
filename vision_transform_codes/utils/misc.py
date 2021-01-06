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


def walk_on_unit_sphere(starting_position, step_angle_radians,
                        num_steps, walk_type='random', additional_params=None):
  """
  Generates a sequence of vectors that traverse the N-dimensional unit sphere.

  We start at starting_position. Then we take num_steps steps along the surface
  of the unit sphere, with each step deviating from the previous step by the
  angle specified by parameter step_angle_radians. We can pick a new direction
  to walk in every step and this is a 'random' walk on the unit sphere.
  Alternatively, we can provide a second vector that, along with the first
  vector, defines a (2D) hyperplane in N-dimensional space and make sure that
  our steps are constrained to this plane (in addition to the unit sphere).
  This traces out a "great circle" of the unit sphere.

  Parameters
  ----------
  starting_position : ndarray(size=(N,))
      A point in N-dimensional euclidean space where we start the walk.
  step_angle_radians : float
      The 'size' of the step in radians. A step angle that is any integer
      multiple of 2pi will bring us back to starting_position.
  num_steps : int
      The number of steps to take
  walk_type : str, optional
      Currently one of {'random', 'great_circle'}. 'random' takes
      randomly-oriented steps, while 'great_circle' traces out a great circle
      of the N-dimensional unit sphere. Default 'random'.
  additional_params : dictionary, optional
      Optional for 'great_circle' walk type, not used for 'random' walk type.
      .. if walk_type == 'great_circle' ..
        'gc_other_vector' : ndarray(size=(N,))
          Another vector which *defines* a particular great circle. If not
          provided we draw one at random.
      Default None.

  Returns
  -------
  steps : ndarray(size=(N, num_steps))
      A sequence of steps defining a walk on the unit sphere
  """
  assert starting_position.ndim == 1
  N = len(starting_position)
  start_pos = starting_position / np.linalg.norm(starting_position)
  assert walk_type in ['random', 'great_circle']
  if walk_type == 'great_circle':
    if additional_params is not None:
      assert 'gc_other_vector' in additional_params
      assert additional_params['gc_other_vector'].ndim == 1
      assert len(additional_params['gc_other_vector']) == N
      gc_other_vector = np.copy(additional_params['gc_other_vector'])
    else:
      gc_other_vector = np.random.randn(N)
      gc_other_vector = gc_other_vector / np.linalg.norm(gc_other_vector)
    # we use np.linalg.qr to compute an orthogonal basis for the plane that
    # defines the great circle
    plane_basis, _ = np.linalg.qr(np.c_[start_pos, gc_other_vector])

  rotation_matrix_2d = np.array([
    [np.cos(step_angle_radians), -np.sin(step_angle_radians)],
    [np.sin(step_angle_radians), np.cos(step_angle_radians)]])
  steps = np.zeros([N, num_steps])
  steps[:, 0] = start_pos
  for step_idx in range(1, num_steps):
    if walk_type == 'random':
      # we pick a different great circle in each step
      gc_other_vector = np.random.randn(N)
      gc_other_vector = gc_other_vector / np.linalg.norm(gc_other_vector)
      plane_basis, _ = np.linalg.qr(np.c_[steps[:, step_idx-1],
                                          gc_other_vector])
    steps[:, step_idx] = np.squeeze(
        np.dot(plane_basis,
          np.dot(rotation_matrix_2d,
            np.dot(plane_basis.T, steps[:, step_idx-1][:, None]))))
  return steps

