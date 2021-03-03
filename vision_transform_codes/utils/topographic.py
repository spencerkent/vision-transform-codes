"""
Utilities for working with topographic versions of sparse coding, specifically
the Laplacian Scale Mixture model of [1]

.. [1] Garrigues, P. & Olshausen, B. (2010). Group Sparse Coding with a
       Laplacian Scale Mixture Prior. NIPS 2010.
"""
import numpy as np


def generate_LSM_topo_connectivity_matrix(rp_topo_size, rp_neighborhood_size,
                                          rp_stride):
  """
  Generates conn. mat for topographic {L}aplacian {S}cale {M}ixture model

  An LSM has a 'second layer' of {r}ate {p}arameters that capture inverse scale
  and project to (potentially multiple) coefficients in the layer below. These
  are denoted as $\lambda_(j)$ in [1]. The topography of the rate parameters
  has shape (k_1, k_2, ..., k_T), which induces a topography on the
  coefficients with shape (s_1, s_2, ..., s_T). Define k as the product of all
  k_t and s as the product of all s_t. These specific values are
  s_t = k_t * m_t, where m_t is the stride in dimension t.

  The connectivity matrix can be used to sum over coefficients in the rate
  parameters's "neighborhood" or (it's transpose) can be used to spread the
  rate parameters to all the affected coefficients, accounting for coefficents
  that get input from multiple rate parameters (this is where the
  neighborhoods overlap).

  Parameters
  ----------
  rp_topo_size : tuple of ints, (k_1, k_2, ..., k_T)
      The topography size for the {r}ate {p}arameters in the LSM model.
      There are T dimensions, each with a size k_t.
  rp_neighborhood_size : tuple of ints (n_1, n_2, ..., n_T)
      In each dimension of the topography, the size of the contiguous
      neighborhood that a scale parameter collects bottom-up input from.
  rp_stride : tuple of ints (m_1, m_2, ..., m_T)
      In each dimension, the stride, in terms of the corresponding coefficient
      topography, of the rate parameters.

  Returns
  -------
  connectivity_matrix : ndarray(float32, size=(s, k))
      For each column, has a 1 for all the coefficients in this column's
      neighborhood.
  """
  assert len(rp_topo_size) == len(rp_neighborhood_size)
  assert len(rp_topo_size) == len(rp_stride)
  topo_ndim = len(rp_topo_size)
  assert all([rp_neighborhood_size[x] > 0 for x in range(topo_ndim)])
  assert all([rp_stride[x] > 0 for x in range(topo_ndim)])
  assert all([rp_stride[x] <= rp_neighborhood_size[x] 
              for x in range(topo_ndim)])

  # iterate over each rate parameter to compute a column of the matrix
  rp_topo_iterator = np.nditer(
      np.zeros(rp_topo_size), flags=['multi_index'], order='C')
  # ^uses the ('C-style') flattening convention to match np.reshape default
  connectivity_matrix = []
  for it in rp_topo_iterator:
    connectivity_tensor = np.zeros(
        np.array(rp_topo_size) * np.array(rp_stride), dtype='float32')
    neighborhood_Nd = []
    for dimension in range(topo_ndim):
      neighborhood_Nd.append(neighborbood(
        rp_topo_iterator.multi_index[dimension],
        rp_stride[dimension], rp_neighborhood_size[dimension],
        rp_topo_size[dimension]))
    connectivity_tensor[np.ix_(*neighborhood_Nd)] = 1
    connectivity_matrix.append(connectivity_tensor.flatten())

  return np.array(connectivity_matrix).T


def neighborbood(ind, stride, n_size, total_num):
  """
  Computes the neighborhood of a particular rate parameter in 1 dimension

  Given the index of the parameter in a sequence of total_num rate
  parameters, each separated by stride, returns the neighborbood, (size n_size)
  of coeffients in the layer below that this rate parameter connects to. We
  wrap around so parameters at the end may have connections to the first couple
  coefficients.

  Parameters
  ----------
  ind : int
      The index of the rate parameter (out of total_num)
  stride : int
      Spacing, in terms of coefficients, between rate parameters
  n_size : int
      The neighborhood size
  total_num : int
      Total number of rate parameters

  Returns
  -------
  neighborhood : ndarray(np.intp, size=(n_size,))
      The neighborhood of the ind^{th} rate parameter.
  """
  return np.array([(ind * stride + x) % (total_num * stride)
                   for x in range(0, n_size)], dtype=np.intp)

def compute_coeff_topo_size(shape, strides):
  return np.array(shape) * np.array(strides)
