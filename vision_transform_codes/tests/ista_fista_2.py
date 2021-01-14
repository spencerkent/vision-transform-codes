"""
Testing convolutional ista_fista analysis_transform.
"""
import _set_the_path

import pickle
import torch

from analysis_transforms.convolutional import ista_fista

from utils import defaults

trn_val_dsets = pickle.load(open(defaults.dataset_directory /
  'vtc_testing/field_white_256x256_w_padding.p', 'rb'))

KSTRIDE=8
PADDING_DIMS=((8, 8), (8, 8))

torch_device = torch.device('cuda:1')

imgs = torch.from_numpy(trn_val_dsets['training']['patches'][0:3]).to(
    torch_device)
# create the dictionary Tensor on the GPU
sparse_coding_dictionary = torch.randn((64, 1, 16, 16), device=torch_device)
# start out the dictionaries with norm 1
sparse_coding_dictionary.div_(torch.squeeze(sparse_coding_dictionary.norm(
  p=2, dim=(1, 2, 3)))[:, None, None, None])

pre_imgs = torch.zeros_like(imgs).copy_(imgs)
pre_dict = torch.zeros_like(sparse_coding_dictionary).copy_(
    sparse_coding_dictionary)

# most vanilla
ista_codes = ista_fista.run(imgs, sparse_coding_dictionary,
                            (KSTRIDE, KSTRIDE), PADDING_DIMS,
                            0.01, 10, variant='ista')
fista_codes = ista_fista.run(imgs, sparse_coding_dictionary,
                             (KSTRIDE, KSTRIDE), PADDING_DIMS,
                             0.01, 10, variant='fista')
# add an early stopping criterion
ista_codes = ista_fista.run(imgs, sparse_coding_dictionary,
                            (KSTRIDE, KSTRIDE), PADDING_DIMS,
                            0.01, 10, variant='ista',
                            early_stopping_epsilon=1e-3)
# add nonnegative-only
ista_codes = ista_fista.run(imgs, sparse_coding_dictionary,
                            (KSTRIDE, KSTRIDE), PADDING_DIMS,
                            0.01, 10, variant='ista',
                            early_stopping_epsilon=1e-3,
                            nonnegative_only=True)
# add hard threshold
ista_codes = ista_fista.run(imgs, sparse_coding_dictionary,
                            (KSTRIDE, KSTRIDE), PADDING_DIMS,
                            0.01, 10, variant='ista',
                            early_stopping_epsilon=1e-3,
                            nonnegative_only=True, hard_threshold=True)
assert torch.allclose(imgs, pre_imgs)
assert torch.allclose(sparse_coding_dictionary, pre_dict)

pre_ista_codes = torch.zeros_like(ista_codes).copy_(ista_codes)
# apply these to existing codes
new_ista_codes = ista_fista.run(imgs, sparse_coding_dictionary,
                                (KSTRIDE, KSTRIDE), PADDING_DIMS,
                                0.01, 10, variant='ista',
                                initial_codes=ista_codes,
                                nonnegative_only=True, hard_threshold=True)
assert torch.allclose(ista_codes, pre_ista_codes) # dont mutate initial_codes
assert not torch.allclose(new_ista_codes, pre_ista_codes)
# TODO: compare convolutional to fully-connected
