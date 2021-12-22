import natsort
import numpy as np
import torch
import os
from torchvision.transforms import transforms
from medpy.io import load


class DatasetSignalAndNoiseSamples(torch.utils.data.Dataset):

    def __init__(self, e, split):
        self.split = split
        self.path_noise_samples = os.path.join(e.path_noise_samples, split)
        self.path_signal_samples = os.path.join(e.path_signal_samples, split)

        self.names_noise_samples = []
        for _, _, f_names in os.walk(self.path_noise_samples):
            for f in f_names:
                self.names_noise_samples.append(f)

        self.names_signal_samples = []
        for _, _, f_names in os.walk(self.path_signal_samples):
            for f in f_names:
                self.names_signal_samples.append(f)

        self.names_noise_samples = natsort.natsorted(self.names_noise_samples)
        self.names_signal_samples = natsort.natsorted(self.names_signal_samples)

        if self.split == 'val' or self.split == 'test':
            if len(self.names_signal_samples) < len(self.names_noise_samples):
                raise Exception("There are less signal samples than noise samples for '" + self.split + "' split. Please generate more signal samples or remove some noise samples from the specified dataset.")

        self.scaling_factor_for_signal_samples = e.scaling_factor_for_signal_samples
        self.random_scaling_for_signal_samples = e.random_scaling_for_signal_samples
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0], std=[e.divisor_for_data_normalization])])

    def __getitem__(self, index):
        name_noise_sample = self.names_noise_samples[index]

        # Use random signal samples for training, but make the selection deterministic for val and test.
        if self.split == 'val' or self.split == 'test':
            index_signal_sample = index
        else:
            index_signal_sample = np.random.randint(len(self.names_signal_samples))

        name_signal_sample = self.names_signal_samples[index_signal_sample]

        noise_sample = load(os.path.join(self.path_noise_samples, name_noise_sample))[0].astype('float32')
        signal_sample = load(os.path.join(self.path_signal_samples, name_signal_sample))[0].astype('float32')

        if self.random_scaling_for_signal_samples:
            if self.split == 'val' or self.split == 'test':
                # decreasing with index so that first val sample (usually the one that is printed) has the highest scaling factor
                current_scaling_factor_for_signal_sample = (1.0 - float(index)/len(self.names_noise_samples)) * self.scaling_factor_for_signal_samples
            else:
                current_scaling_factor_for_signal_sample = (1.0 - np.random.rand()) * self.scaling_factor_for_signal_samples

        else: # Deterministic scaling factor
            current_scaling_factor_for_signal_sample = self.scaling_factor_for_signal_samples

        signal_sample = signal_sample * current_scaling_factor_for_signal_sample

        noisy_signal = self.transform(signal_sample + noise_sample)
        noise_sample = self.transform(noise_sample)
        # true_signal = self.transform(data_signal)
        return noisy_signal, noise_sample, name_noise_sample, name_signal_sample

    def __len__(self):
        """Return the total number of images."""
        return len(self.names_noise_samples)
