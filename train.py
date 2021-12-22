import math
import pickle
import time
import torch
import numpy as np
from datasets.dataset_invivo_sinograms import DatasetInvivoSinograms
from datasets.dataset_signal_and_noise_samples import DatasetSignalAndNoiseSamples
from loggers.epoch_loss_accumulator import EpochLossAccumulator
from loggers.print_epoch_loss import print_epoch_loss
from models.network_denoising import DenoisingNet
from torch.utils.tensorboard import SummaryWriter
import os
from set_locals.set_local_experiment_train_val import set_local_experiment_train_val
from utils import stopwatch
from utils.custom_loss import custom_loss
from utils.environment_check import environment_check
from utils.get_optimizer import get_optimizer
from utils.get_output_folders import get_output_folders_for_train_val
from utils.get_scheduler import get_scheduler
from utils.visualize_denoising_of_sinogram import visualize_denoising_of_sinogram
from shutil import copyfile

def main():
    # Startup the experiment
    e = set_local_experiment_train_val()
    use_cuda, device, num_workers = environment_check(e.gpu_index)
    tensorboard_output_dir, model_checkpoint_output_dir = get_output_folders_for_train_val(
        e.save_path, e.experiment_name, create_missing_folders=True)
    writer = SummaryWriter(tensorboard_output_dir)
    print("Writing tensorboard logs to '" + tensorboard_output_dir + "'")

    # Save experiment object and parameters file
    pickle.dump(e, open(os.path.join(model_checkpoint_output_dir, 'experiment_train_val.pickle'), "wb"))
    copyfile(os.path.join('set_locals', 'set_local_experiment_train_val.py'), os.path.join(model_checkpoint_output_dir, 'set_local_experiment_train_val.py'))


    # Define data sources
    params_dataloader_train = {'batch_size': e.batch_size_train, 'shuffle': True, 'num_workers': 3, 'drop_last': False}
    dataloader_train = torch.utils.data.DataLoader(dataset=DatasetSignalAndNoiseSamples(e, 'train'), **params_dataloader_train)

    params_dataloader_val = {'batch_size': e.batch_size_val, 'shuffle': False, 'num_workers': 3, 'drop_last': False}
    dataloader_val = torch.utils.data.DataLoader(dataset=DatasetSignalAndNoiseSamples(e, 'val'), **params_dataloader_val)

    print('The number of training images = %d' % len(dataloader_train.dataset))
    print('The number of validation images = %d' % len(dataloader_val.dataset))

    if e.path_test_samples:
        params_dataloader_test = {'batch_size': 1, 'shuffle': False, 'num_workers': 3, 'drop_last': False}
        dataloader_test = torch.utils.data.DataLoader(dataset=DatasetInvivoSinograms(e.path_test_samples, e.divisor_for_data_normalization), **params_dataloader_test)
        print('The number of test images = %d' % len(dataloader_test.dataset))


    # Set up network and learning tools
    network = DenoisingNet(e)
    if use_cuda:
        network.cuda(device=device)
    network.float()

    optimizer = get_optimizer(network.parameters(), e)
    scheduler = get_scheduler(optimizer, e)

    global best_validation_loss
    best_validation_loss = math.inf


    # Do the training
    network.train()
    for id_epoch in range(e.num_epochs):
        train(e, network, dataloader_train, optimizer, scheduler, device, writer, id_epoch)
        val(e, network, dataloader_val, optimizer, scheduler, device, writer, id_epoch, model_checkpoint_output_dir)
        if e.path_test_samples:
            test(e, network, dataloader_test, device, writer, id_epoch)

def train(e, network, dataloader_train, optimizer, scheduler, device, writer, id_epoch):
    # global id_batch, noisy_signal, noise_sample, name_noise_sample, name_signal_sample, output, loss
    time_trn_one_epoch = time.time()
    train_loss_accumulator = EpochLossAccumulator()
    for id_batch, (noisy_signal, noise_sample, name_noise_sample, name_signal_sample) in enumerate(dataloader_train):
        optimizer.zero_grad()
        # --- Forward pass
        output = network(noisy_signal.to(device).float())
        # --- Calculate loss
        loss = custom_loss(output.float(), noise_sample.to(device).float(), e.loss_nickname)
        # --- Backward pass
        loss.backward()
        # --- Optimize
        optimizer.step()
        # --- Keep track of calculating and displaying the score
        train_loss_accumulator.update_losses(e.batch_size_train, loss.item())
        #print_batch_loss(id_epoch, 'Trn', train_loss_accumulator, time_trn_one_epoch, id_batch, len(dataloader_train))
        # --- Save first image of first batch to Tensorboard
        if id_batch == 0:
            fig = visualize_denoising_of_sinogram(np.squeeze(noisy_signal.detach().cpu().numpy()[0])*e.divisor_for_data_normalization,
                                                  np.squeeze(output.detach().cpu().numpy()[0])*e.divisor_for_data_normalization,
                                                  name_signal_sample[0],
                                                  np.squeeze(noise_sample.detach().cpu().numpy()[0])*e.divisor_for_data_normalization,
                                                  name_noise_sample[0],
                                                  clim=e.clim_printed_sinograms)
            writer.add_figure('0_Trn', fig, global_step=id_epoch)
    # --- Log to Console and Tensorboard
    current_train_loss = train_loss_accumulator.get_epoch_loss()
    print_epoch_loss('Trn', id_epoch, current_train_loss, time_trn_one_epoch)
    writer.add_scalar('0_training_loss', current_train_loss, id_epoch)
    # --- Learning rate evolution
    scheduler.step()

def val(e, network, dataloader_val, optimizer, scheduler, device, writer, id_epoch, model_checkpoint_output_dir):
    # global id_batch, noisy_signal, noise_sample, name_noise_sample, name_signal_sample, output, loss, best_validation_loss
    global best_validation_loss
    network.eval()
    val_loss_accumulator = EpochLossAccumulator()
    time_val_one_epoch = time.time()
    with torch.no_grad():
        for id_batch, (noisy_signal, noise_sample, name_noise_sample, name_signal_sample) in enumerate(dataloader_val):
            # --- Forward pass
            output = network(noisy_signal.to(device).float())
            # --- Loss
            loss = custom_loss(output.float(), noise_sample.to(device).float(), e.loss_nickname)
            val_loss_accumulator.update_losses(e.batch_size_val, loss.item())
            #print_batch_loss(id_epoch, 'Val', val_loss_accumulator, time_val_one_epoch, id_batch, len(dataloader_val))
            # --- Save image to Tensorboard
            index_for_printing = name_signal_sample.index(e.name_printed_signal_val) if e.name_printed_signal_val in name_signal_sample else -1
            if index_for_printing > -1:
                fig = visualize_denoising_of_sinogram(np.squeeze(noisy_signal.detach().cpu().numpy()[index_for_printing])*e.divisor_for_data_normalization,
                                                      np.squeeze(output.detach().cpu().numpy()[index_for_printing])*e.divisor_for_data_normalization,
                                                      name_signal_sample[index_for_printing],
                                                      np.squeeze(noise_sample.detach().cpu().numpy()[index_for_printing])*e.divisor_for_data_normalization,
                                                      name_noise_sample[index_for_printing],
                                                      clim=e.clim_printed_sinograms)
                writer.add_figure('1_Val', fig, global_step=id_epoch)

        # --- Log to Console and Tensorboard
        current_val_loss = val_loss_accumulator.get_epoch_loss()
        print_epoch_loss('Val', id_epoch, current_val_loss, time_val_one_epoch)
        writer.add_scalar('1_validation_loss', current_val_loss, id_epoch)
        writer.flush()
        # --- Save current network checkpoint if new minimal validation loss is found
        if current_val_loss < best_validation_loss:
            best_validation_loss = current_val_loss
            torch.save({
                'epoch': id_epoch + 1,
                'network_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()},
                os.path.join(model_checkpoint_output_dir, 'model_min_val_loss.pt'))

def test(e, network, dataloader_test, device, writer, id_epoch):
    network.eval()
    time_test_one_epoch = time.time()
    with torch.no_grad():
        for id_batch, (noisy_signal, name_noisy_signal) in enumerate(dataloader_test):
            index_for_printing = name_noisy_signal.index(e.name_printed_signal_test) if e.name_printed_signal_test in name_noisy_signal else -1
            if index_for_printing > -1:
                # --- Forward pass
                output = network(noisy_signal.to(device).float())
                # --- Save image to Tensorboard
                fig = visualize_denoising_of_sinogram(np.squeeze(noisy_signal.detach().cpu().numpy()[index_for_printing])*e.divisor_for_data_normalization,
                                                      np.squeeze(output.detach().cpu().numpy()[index_for_printing])*e.divisor_for_data_normalization,
                                                      name_noisy_signal[index_for_printing],
                                                      clim=e.clim_printed_sinograms)
                writer.add_figure('2_Tst', fig, global_step=id_epoch)
                break
        print('[Epoch: {}, \t {}] (Stopwatch: {})'.format(id_epoch, 'Tst', stopwatch.stopwatch(time.time(), time_test_one_epoch)))
        writer.flush()

if __name__ == '__main__':
    main()
