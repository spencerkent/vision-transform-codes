"""
Utilities for implementing the JPEG standard.

This is mostly related to the source coding scheme that JPEG uses. As of
Sep 17, 2019 there is a gap between rate distortion performance of this
implementation and the implementation used by FFMPEG. Still tracking down
the discrepancy.
"""
from collections import defaultdict
import numpy as np
from heapq import heappush, heappop, heapify

from . import matrix_zigzag


#########################
# Couple helper functions
#########################
def get_jpeg_quant_hifi_binwidths():
  # these are recommended binwidths for each dimension when the images are
  # in the range [0, 255]. You will want to rescale these to reflect the range
  # of your own data if it does not match this...
  uint8_8x8_quant_binwidths = np.array([[16,11,10,16,24,40,51,61],
                                        [12,12,14,19,26,58,60,55],
                                        [14,13,16,24,40,57,69, 56],
                                        [14,17,22,29,51,87,80,62],
                                        [18,22,37,56,68,109,103,77],
                                        [24,35,55,64,81,104,113,92],
                                        [49,64,78,87,103,121,120,101],
                                        [72,92,95,98,112,100,103,99]])
  return matrix_zigzag.zigzag(uint8_8x8_quant_binwidths)


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
#############################
# End couple helper functions
#############################


def generate_ac_dc_huffman_tables(all_assignment_inds, inds_of_zero_valued_cw):
  """
  Generates a huffman table for the JPEG runlength symbols

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
  huff_table_ac : dictionary(str)
      Just a lookup table that converts symbol 1 in the JPEG runlength encoding
      (an 8-bit hex string) into a binary codeword
  huff_table_dc : dictionary(str)
      Just a lookup table that converts dc 'category' param in the JPEG
      runlength encoding (a hex string) into a binary codeword.
  """
  counts_ac_symbs = defaultdict(int)
  counts_dc_symbs = defaultdict(int)
  for data_pt_idx in range(all_assignment_inds.shape[0]):
    ac_symbs, dc_symb = generate_jpg_binary_stream(
        all_assignment_inds[data_pt_idx], inds_of_zero_valued_cw,
        only_get_huffman_symbols=True)
    for x in ac_symbs:
      counts_ac_symbs[x] += 1
    counts_dc_symbs[dc_symb] += 1

  # we have an issue of runlength symbols not seen in the training set.
  # we can't afford to send any runlength symbol incorrectly because it can
  # screw up the whole image. Thus, we have to make sure that any of the
  # reasonably possible 8-bit symbols is in this dictionary.
  # We'll just add them as if we only saw one occurence of them.
  for first_4_bits in range(15):
    for second_4_bits in range(10):
      #^ according to the JPEG standard we don't have to worry about ac values
      # that take more then 10 bits to encode...
      rl_str = hex(first_4_bits)[2:] + hex(second_4_bits)[2:]
      if rl_str not in counts_ac_symbs:
        counts_ac_symbs[rl_str] = 1
  # we're going to assume DC values never get bigger than 255*64 = 16320
  for bitnum in range(1, 15):
    if hex(bitnum)[2:] not in counts_dc_symbs:
      counts_dc_symbs[hex(bitnum)[2:]] = 1

  huff_table_ac = compute_huffman_table(counts_ac_symbs)
  huff_table_dc = compute_huffman_table(counts_dc_symbs)

  return huff_table_ac, huff_table_dc


def jpg_coeff_to_binstr(decimal_number):
  """
  Assigns jpeg coefficients a binary str using a 1's-complement-like encoding
  """
  if decimal_number == 0:
    return ''  # the only place this will come up is if the DC coeff is
               # exactly zero. The decoder will catch this and insert a 0

  elif decimal_number > 0:
    return "{0:b}".format(decimal_number)

  else:
    # negative numbers require a little more love
    temp_string = "{0:b}".format(-decimal_number)
    str2 = str("")
    for i in range(len(temp_string)):
      if temp_string[i] == '0':
        str2 += '1'
      if temp_string[i] == '1':
        str2 += '0'
    assert len(str2) == len(temp_string)
    return str2


def generate_jpg_binary_stream(assignment_inds, inds_of_zero_valued_cw,
                               only_get_huffman_symbols=True,
                               huffman_table_ac=None, huffman_table_dc=None):
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
  huffman_table_ac : dictionary(str)
      Just a lookup table that converts symbol 1 in the JPEG runlength encoding
      (an 8-bit hex string) into a binary codeword
  huffman_table_dc : dictionary(str)
      Just a lookup table that converts dc 'category' param in the JPEG
      runlength encoding (a hex string) into a binary codeword.

  Returns
  -------
  full_binary_stream : str  (if only_get_huffman_symbols=False)
      The actual binary stream for this data point
  runlength_stream : list(str)  (if only_get_huffman_symbols=True)
      The runlength symbol sequence
  dc_len_str : str  (if only_get_huffman_symbols=True)
      The single string symbol giving the 'category' value for the dc component
  """

  if not only_get_huffman_symbols:
    assert (huffman_table_dc is not None) and (huffman_table_ac is not None)

  # our convention outside the scope of JPEG is that assignment indexes are
  # always nonnegative. However, the JPEG source coding scheme assumes the
  # index of the zero-valued codeword is 0, with all indexes less than this
  # index negative. We just make this reassignment to make it match original
  # JPEG.
  jpeg_quant_assigns = assignment_inds - inds_of_zero_valued_cw

  idx_last_nonzero = -1
  for i, elem in enumerate(jpeg_quant_assigns):
    if elem != 0:
      idx_last_nonzero = i

  runlength_stream = []  # this contains 8-bit values with upper 4 bits for the
                         # runlength and lower 4 bits for the binary size of
                         # the forthcoming raw value. This 8-bit value is
                         # what gets coded by a Huffman code.

  ac_value_stream = []  # these are the binary representations of the raw AC
                        # values. Weirdly, they don't get Huffman coded.
  previous_zeros = 0
  for code_idx in range(1, idx_last_nonzero + 1):
    value = jpeg_quant_assigns[code_idx]
    if previous_zeros > 15:
      # we write a special character indicating 16 zeros
      runlength_stream.append('f0')
      ac_value_stream.append(jpg_coeff_to_binstr(0))
      previous_zeros = 0
    if value != 0:
      ac_value_str = jpg_coeff_to_binstr(value)
      runlength_stream.append(
          hex(previous_zeros)[2:] + hex(len(ac_value_str))[2:])
      ac_value_stream.append(ac_value_str)
      previous_zeros = 0
    else:
      previous_zeros += 1
  # append the EOB symbol
  runlength_stream.append('00')
  assert len(runlength_stream) == len(ac_value_stream) + 1

  dc_value_str = jpg_coeff_to_binstr(jpeg_quant_assigns[0])
  if len(dc_value_str) == 0:  # the actual DC value was zero
    dc_len_str = '-' # this is a special character
  else:
    dc_len_str = hex(len(dc_value_str))[2:]

  if only_get_huffman_symbols:
    # we go no further, this is just being used to collect symbol statistics
    # for generating the Huffman tables at training time
    return runlength_stream, dc_len_str

  # we assign Huffman codewords for the runlength stream
  binary_runlength_stream = [huffman_table_ac[x] for x in runlength_stream]

  # we assign Huffman codewords for the bitlength param
  binary_dc_len_codeword = huffman_table_dc[dc_len_str]

  full_binary_stream = ''
  # append the runlength AC part
  for ac_stream_idx in range(len(ac_value_stream)):
    full_binary_stream += binary_runlength_stream[ac_stream_idx]
    full_binary_stream += ac_value_stream[ac_stream_idx]
  full_binary_stream += binary_runlength_stream[-1]  # codeword for EOB

  # append the DC part at the end
  full_binary_stream += binary_dc_len_codeword
  full_binary_stream += dc_value_str

  return full_binary_stream
