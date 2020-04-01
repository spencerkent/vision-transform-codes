"""
Just some basic defaults to keep in one place
"""
from pathlib import Path

logging_directory = Path(
    '/media/expansion1/spencerkent/logfiles/vision_transform_codes')
dataset_directory = Path('/media/expansion1/spencerkent/Datasets')

raw_data_filepaths = {
    'Field_NW': dataset_directory / 'Field_natural_images/unwhitened.mat',
    'vanHateren': dataset_directory / 'vanHaterenNaturalScenes/curated_by_dylan_paiton.h5',
    'Kodak_BW': dataset_directory / 'Kodak/kodak_full_images_training.p'}

