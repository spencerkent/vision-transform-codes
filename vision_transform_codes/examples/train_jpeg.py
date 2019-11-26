"""
Demonstrates using the DCT and spitting out JPEG source codes
"""
import _set_the_path

import argparse
import pickle
import numpy as np
from matplotlib import pyplot as plt
import torch

from analysis_transforms import invertible_linear
from utils.image_processing import create_patch_training_set
from utils.plotting import display_dictionary
from utils.plotting import display_codes
from utils.plotting import display_code_marginal_densities
from utils.matrix_zigzag import zigzag

RUN_IDENTIFIER = 'test_JPEG'

NUM_IMAGES_TRAIN = 1000000
PATCH_HEIGHT = 8
PATCH_WIDTH = 8
assert PATCH_HEIGHT == 8  # I haven't tested the source coding with larger patches
assert PATCH_WIDTH == 8

CODE_SIZE = PATCH_HEIGHT * PATCH_WIDTH

# Arguments for dataset and logging
parser = argparse.ArgumentParser()
parser.add_argument("data_id",
    help="Name of the dataset (currently allowable: " +
         "Field_NW_whitened, Field_NW_unwhitened, vanHateren, Kodak)")
parser.add_argument("data_filepath", help="The full path to dataset on disk")
parser.add_argument("dct_matrix_filepath",
                    help="This is where the DCT matrix is stored on disk")
script_args = parser.parse_args()

torch_device = torch.device('cuda:1')
torch.cuda.set_device(1)
# otherwise can put on 'cuda:0' or 'cpu'

# manually create large training set with one million whitened patches
one_mil_image_patches = create_patch_training_set(
    ['patch', 'center_each_component'], (PATCH_HEIGHT, PATCH_WIDTH),
    NUM_IMAGES_TRAIN, 1, edge_buffer=5, dataset=script_args.data_id,
    datasetparams={'filepath': script_args.data_filepath,
                   'exclude': []})['batched_patches']

#################################################################
# save these to disk if you want always train on the same patches
# or if you want to speed things up in the future
#################################################################
# pickle.dump(one_mil_image_patches, open('/media/expansion1/spencerkent/Datasets/Field_natural_images/one_million_patches_whitened_June25.p', 'wb'))

# one_mil_image_patches = pickle.load(open(
#     '/media/expansion1/spencerkent/Datasets/Field_natural_images/one_million_patches_whitened_June25.p', 'rb')).astype('float32')

# we are going to 'unbatch' them because DCT will train on the whole dataset
# at once
image_patches_gpu = torch.from_numpy(
    one_mil_image_patches.transpose((0, 2, 1)).reshape(
      (-1, PATCH_HEIGHT*PATCH_WIDTH)).T).to(torch_device)

dct_matrix_zigzag_order = pickle.load(
    open(script_args.dct_matrix_filepath, 'rb')).astype('float32')
reordered_inds = zigzag(
    np.arange(64, dtype='int').reshape((8, 8))).astype('int')
dct_matrix = dct_matrix_zigzag_order[:, reordered_inds]

torch_dct_matrix = torch.from_numpy(dct_matrix).to(torch_device)
codes = invertible_linear.run(image_patches_gpu, torch_dct_matrix,
                              orthonormal=True).cpu().numpy()
# display_dictionary(dct_matrix, (8, 8), renormalize=False)
# display_codes(codes[:, 0:20], 'Example of DCT codes for random patches')
# display_code_marginal_densities(codes, 100, 'marginal dct coeff densities')
# plt.show()


# an example of how we can take quantize these values and spit out a source code
from utils.jpeg import get_jpeg_quant_hifi_binwidths
from utils.jpeg import generate_ac_dc_huffman_tables
from utils.jpeg import generate_jpg_binary_stream
from utils.quantization import cbook_inds_of_zero_pts

sys.path.insert(0, '/home/spencerkent/Projects/generalized-lloyd-quantization/')
from generalized_lloyd_quantization import null_uniform as uniform_quant

jpeg_hifi_binwidths = get_jpeg_quant_hifi_binwidths()
# for the Field dataset, code variance is much smaller so I'm going to make
# the quant binwidths much smaller too
codebook = []
all_assignments = np.zeros(codes.T.shape, dtype='int')
for coeff_idx in range(64):
  apts, assignments, _, _ = uniform_quant.compute_quantization(
      codes[coeff_idx, :], jpeg_hifi_binwidths[coeff_idx],
      placement_scheme='on_zero')
  codebook.append(apts)
  all_assignments[:, coeff_idx] = assignments

zero_inds = cbook_inds_of_zero_pts(codebook)

print('Generating huffman tables. You can save these to disk')
huff_tab_ac, huff_tab_dc = generate_ac_dc_huffman_tables(all_assignments,
                                                         zero_inds)

# now given these huffman tables, we can generate jpeg bitstreams for any new
# image. Here I'll just apply it to training patches for illustration.
print('This is the jpeg bitstream for a patch')
print(generate_jpg_binary_stream(all_assignments[0], zero_inds,
                                 only_get_huffman_symbols=False,
                                 huffman_table_ac=huff_tab_ac,
                                 huffman_table_dc=huff_tab_dc))
