"""
Just some basic defaults to keep in one place
"""
from pathlib import Path
import numpy as np

logging_directory = Path(
    '/media/expansion1/spencerkent/logfiles/vision_transform_codes')
dataset_directory = Path('/media/expansion1/spencerkent/Datasets')

raw_data_filepaths = {
    'Field_NW': dataset_directory / 'Field_natural_images/unwhitened.mat',
    'vanHateren': dataset_directory / 'vanHaterenNaturalScenes/curated_by_dylan_paiton.h5',
    'Kodak_BW': dataset_directory / 'Kodak/kodak_full_images_training.p'}

# just useful for sampling a gabor without having to specify all the parameters
gabor_params = {'patch_size': (16, 16), 'gabor_parameters': {
  'orientation': np.pi/4, 'envelope_width': 3, 'envelope_aspect': 0.5,
  'frequency': 1/4, 'phase': 0, 'position_yx': (0, 0)}}
