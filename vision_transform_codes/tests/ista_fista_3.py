"""
Testing the subspace variant of fully-connected ISTA/FISTA
"""
import _set_the_path

import pickle
import numpy as np
import torch

from analysis_transforms.fully_connected import subspace_ista_fista

from utils import defaults

trn_val_dsets = pickle.load(open(defaults.dataset_directory /
  'vtc_testing/field_white_16x16.p', 'rb'))

torch_device = torch.device('cuda:1')

imgs = torch.from_numpy(trn_val_dsets['training']['patches']).to(torch_device)
# create the dictionary Tensor on the GPU
sparse_coding_dictionary = torch.randn((256, 256), device=torch_device)
# start out the dictionaries with norm 1
sparse_coding_dictionary.div_(
    sparse_coding_dictionary.norm(p=2, dim=1)[:, None])

pre_imgs = torch.zeros_like(imgs).copy_(imgs)
pre_dict = torch.zeros_like(sparse_coding_dictionary).copy_(
    sparse_coding_dictionary)

g_assignments = np.array_split(np.arange(256), 256//4)

# most vanilla
ista_codes = subspace_ista_fista.run(imgs, sparse_coding_dictionary,
      g_assignments, 0.01, 100, variant='ista')
fista_codes = subspace_ista_fista.run(imgs, sparse_coding_dictionary,
      g_assignments, 0.01, 100, variant='fista')
# add an early stopping criterion
ista_codes = subspace_ista_fista.run(imgs, sparse_coding_dictionary,
      g_assignments, 0.01, 100, variant='ista',
      early_stopping_epsilon=1e-3)
assert torch.allclose(imgs, pre_imgs)
assert torch.allclose(sparse_coding_dictionary, pre_dict)

pre_ista_codes = torch.zeros_like(ista_codes).copy_(ista_codes)
# apply these to existing codes
new_ista_codes = subspace_ista_fista.run(imgs, sparse_coding_dictionary,
    g_assignments, 0.01, 100, variant='ista', initial_codes=ista_codes)

assert torch.allclose(ista_codes, pre_ista_codes) # dont mutate initial_codes
assert not torch.allclose(new_ista_codes, pre_ista_codes)
