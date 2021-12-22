import natsort
import numpy as np
import torch
import os
from torchvision.transforms import transforms
import re
from PIL import Image
from medpy.io import load


class DatasetInvivoSinograms(torch.utils.data.Dataset):

    def __init__(self, path_test_samples, divisor_for_data_normalization, regex_fullmatch_for_filenames='.*'):
        self.path_sinograms = os.path.join(path_test_samples, 'test')

        self.names_sinograms = []
        for _, _, f_names in os.walk(self.path_sinograms):
            for f in f_names:
                if re.fullmatch(regex_fullmatch_for_filenames, f):
                    self.names_sinograms.append(f)

        self.names_sinograms = natsort.natsorted(self.names_sinograms)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0], std=[divisor_for_data_normalization])])

    def __getitem__(self, index):
        name_sinogram = self.names_sinograms[index]
        sinogram = load(os.path.join(self.path_sinograms, name_sinogram))[0].astype('float32')

        sinogram = self.transform(sinogram)
        return sinogram, name_sinogram

    def __len__(self):
        return len(self.names_sinograms)
