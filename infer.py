import pickle
import time
import numpy as np
import torch
import os
from datasets.dataset_invivo_sinograms import DatasetInvivoSinograms
from models.network_denoising import DenoisingNet
from torch.utils.tensorboard import SummaryWriter
from set_locals.set_local_experiment_infer import set_local_experiment_infer
from utils.environment_check import environment_check
from utils.get_output_folders import get_output_folders_for_train_val, get_output_folder_for_infer
from medpy.io import save

if __name__ == '__main__':
    e_infer = set_local_experiment_infer()

    e_train_val = pickle.load(open(os.path.join(e_infer.path_experiment_train_val_and_weights, 'experiment_train_val.pickle'), 'rb'))

    use_cuda, device, num_workers = environment_check(e_infer.gpu_index_for_inference)
    experiment_base_path, denoised_sinogrmas_path = get_output_folder_for_infer(e_infer.save_path_infer, e_train_val.experiment_name)

    # Define network and load weights
    network = DenoisingNet(e_train_val)
    checkpoint = torch.load(os.path.join(e_infer.path_experiment_train_val_and_weights, 'model_min_val_loss.pt'), map_location=device)
    network.load_state_dict(checkpoint['network_state_dict'])
    network.eval()
    if use_cuda:
        network.cuda()

    # Load test dataset for inference
    params_dataloader_test = {'batch_size': 1, 'shuffle': False, 'num_workers': 3, 'drop_last': False}
    dataloader_test = torch.utils.data.DataLoader(
        dataset=DatasetInvivoSinograms(e_infer.path_noisy_input_sinograms,
                                       e_train_val.divisor_for_data_normalization,
                                       regex_fullmatch_for_filenames=e_infer.regex_fullmatch_for_filenames),
        **params_dataloader_test)
    print('The number of test images = %d' % len(dataloader_test.dataset))

    time_infer_start = time.time()
    with torch.no_grad():
        for id_batch, (noisy_signal, name_noisy_signal) in enumerate(dataloader_test):
            # --- Forward pass
            output = network(noisy_signal.to(device).float())
            denoised_signal = noisy_signal - output.cpu()

            noisy_signal = np.squeeze(noisy_signal.numpy()) * e_train_val.divisor_for_data_normalization
            denoised_signal = np.squeeze(denoised_signal.numpy()) * e_train_val.divisor_for_data_normalization

            # Note: Don't save noisy test sinograms because they are not altered by the network
            save(denoised_signal, os.path.join(denoised_sinogrmas_path, name_noisy_signal[0] + '.nii'))
            print('Saved denoised sinogram of "%s".' % name_noisy_signal[0])
