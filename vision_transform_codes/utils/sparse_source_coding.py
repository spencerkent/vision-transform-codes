"""
Implements source codes for sparse data
"""
import numpy as np
from heapq import heappush, heappop, heapify
from collections import defaultdict

# This is a really simple implementation of a Huffman code that I stole
# from here: http://rosettacode.org/wiki/Huffman_coding#Python
def compute_huffman_table(symb2freq):
  """Huffman encode the given dict mapping symbols to weights"""
  heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
  heapify(heap)
  while len(heap) > 1:
    lo = heappop(heap)
    hi = heappop(heap)
    for pair in lo[1:]:
      pair[1] = '0' + pair[1]
    for pair in hi[1:]:
      pair[1] = '1' + pair[1]
    heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
  return dict(sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p)))


def generate_dense_idx_huffman_tables(all_relative_to_zero_assignments,
                                      vector_code=False):
  counts_cw_idx_symbs = defaultdict(int)
  for x in np.nditer(all_relative_to_zero_assignments):
    counts_cw_idx_symbs[str(x)] += 1

  # for any nz index we didn't see we'll act like we got one sample so that
  # it'll be in the codebook
  # same thing for codeword index, but instead I'm going to limit it to 200
  # quantization points (way overkill of course)
  if vector_code:
    for possible_cw_inds in range(0, 10000):
      if str(possible_cw_inds) not in counts_cw_idx_symbs:
        counts_cw_idx_symbs[str(possible_cw_inds)] = 1
  else:
    for possible_cw_inds in range(-200, 200):
      if str(possible_cw_inds) not in counts_cw_idx_symbs:
        # print('not in Htable: ', )
        counts_cw_idx_symbs[str(possible_cw_inds)] = 1

  huff_table_cw_idx = compute_huffman_table(counts_cw_idx_symbs)

  return huff_table_cw_idx


def generate_idx_msg_huffman_tables(all_assignment_inds,
                                    inds_of_zero_valued_cw):
  """
  Generates a huffman table for the first and second halves of the idx msg

  Parameters
  ----------
  all_assignment_inds : ndarray(int) size=(D, s)
      The set of codeword indeces for a training set. D is the size of the
      training set and s is the size of the code
  inds_of_zero_valued_cw : ndarray(int) size=(s, )
      The indeces of the scalar codeword in each dimension that
      is precisely zero.

  Returns
  -------
  huff_table_nz_idx : dictionary(str)
      Just a lookup table that converts symbols in the first half of the
      index message into binary string
  huff_table_cw_idx : dictionary(str)
      Just a lookup table that converts symbols in the second half of the
      index message into binary string
  """
  counts_nz_idx_symbs = defaultdict(int)
  counts_cw_idx_symbs = defaultdict(int)
  for data_pt_idx in range(all_assignment_inds.shape[0]):
    nz_idx_symbs, cw_idx_symbs = generate_idx_msg_binary_stream(
        all_assignment_inds[data_pt_idx], inds_of_zero_valued_cw,
        only_get_huffman_symbols=True)
    for x in nz_idx_symbs:
      counts_nz_idx_symbs[x] += 1
    for x in cw_idx_symbs:
      counts_cw_idx_symbs[x] += 1
  assert '0' not in counts_cw_idx_symbs  # we shouldn't be sending any zeros

  # for any nz index we didn't see we'll act like we got one sample so that
  # it'll be in the codebook
  for possible_nz_vals in range(all_assignment_inds.shape[1]):
    if str(possible_nz_vals) not in counts_nz_idx_symbs:
      counts_nz_idx_symbs[str(possible_nz_vals)] = 1
  # same thing for codeword index, but instead I'm going to limit it to 200
  # quantization points (way overkill of course)
  for possible_cw_inds in range(-200, 200):
    if str(possible_cw_inds) not in counts_cw_idx_symbs:
      counts_cw_idx_symbs[str(possible_cw_inds)] = 1


  huff_table_nz_idx = compute_huffman_table(counts_nz_idx_symbs)
  huff_table_cw_idx = compute_huffman_table(counts_cw_idx_symbs)

  return huff_table_nz_idx, huff_table_cw_idx


def generate_idx_msg_binary_stream(assignment_inds, inds_of_zero_valued_cw,
                                   only_get_huffman_symbols=True,
                                   huffman_table_nz_inds=None,
                                   huffman_table_nz_codeword_inds=None):
  """
  Generates a binary stream that represents a single data point

  Parameters
  ----------
  assignment_inds : ndarray(int) size=(s,)
      The codeword indeces for a single datapoint
  inds_of_zero_valued_cw : ndarray(int) size=(s, )
      The indeces of the scalar codeword in each dimension that
      is precisely zero.
  only_get_huffman_symbols : bool, optional
      This will just return symbols rather than a binary stream if this
      parameter is set. it's just a convenience for the
      huffman table generation function. Whenever you call this on test data
      this should be False so that you get back a binary stream.
  huffman_table_nz_inds : dictionary(str)
      Just a lookup table that converts symbols in the first half of the
      index message into binary string
  huffman_table_nz_codeword_inds : dictionary(str)
      Just a lookup table that converts symbols in the second half of the
      index message into binary string

  Returns
  -------
  full_binary_stream : str  (if only_get_huffman_symbols=False)
      The actual binary stream for this data point
  msg_nz_inds : list(str)  (if only_get_huffman_symbols=True)
      The first half of the message, in discrete symbols
  msg_nz_codewords : list(str)  (if only_get_huffman_symbols=True)
      The second half of the message, in discrete symbols
  """
  assert assignment_inds.ndim == 1
  if not only_get_huffman_symbols:
    assert ((huffman_table_nz_inds is not None) and
            (huffman_table_nz_codeword_inds is not None))

  # we want nonzero codewords in each dimension to have the same symbol
  # if they are in the same offset from the zero bin
  # so that the Huffman code will take advantage of more probably values
  # like -1, and 1, so we're going to use the same approach as we do
  # in jpeg where we make all the centered on zero
  jpeg_quant_assigns = assignment_inds - inds_of_zero_valued_cw

  temp = np.where(jpeg_quant_assigns != 0)[0]
  msg_nz_inds = temp.astype('str')
  msg_nz_codewords = jpeg_quant_assigns[temp].astype('str')

  # append the special EOB character to the nz_inds message
  msg_nz_inds = np.append(msg_nz_inds, '*')

  if only_get_huffman_symbols:
    return msg_nz_inds, msg_nz_codewords

  # we assign Huffman codewords for the nonzero index stream
  binary_nz_idx_stream = [huffman_table_nz_inds[x] for x in msg_nz_inds]

  # we assign Huffman codewords for the codeword index stream
  binary_nz_codeword_stream = [huffman_table_nz_codeword_inds[x]
                               for x in msg_nz_codewords]
  full_binary_stream = ''
  for nz_idx_huff in binary_nz_idx_stream:
    full_binary_stream += nz_idx_huff
  for nz_cw_huff in binary_nz_codeword_stream:
    full_binary_stream += nz_cw_huff

  return full_binary_stream


def decode_idx_msg(inds_of_zero_valued_cw, msg_nz_inds, msg_nz_codeword_inds):
  """
  Generates the full set of codeword indeces from the index message

  Parameters
  ----------
  inds_of_zero_valued_cw : ndarray(int)
      The indeces of the scalar codeword in each dimension that
      is precisely zero.
  msg_nz_inds : ndarray(str)
      The first part of the code for each datapoint - indexes of the quantized
      code that are nonzero.
  msg_nz_codeword_inds : ndarray(str)
      The corresponding (jpeg-shifted) codeword indeces for each of the
      nonzero coefficients
  """
  int_msg_nz_inds = msg_nz_inds[:-1].astype('int')
  int_msg_nz_codeword_inds = msg_nz_codeword_inds.astype('int')
  original_assignments = np.copy(inds_of_zero_valued_cw)
  original_assignments[int_msg_nz_inds] = (
      int_msg_nz_codeword_inds + inds_of_zero_valued_cw[int_msg_nz_inds])
  return original_assignments
