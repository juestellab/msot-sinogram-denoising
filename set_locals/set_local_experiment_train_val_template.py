# ****************************************************************
# *** HOWTO
# **************************************************************** 
# 0) Do not modify this template file "set_local_experiment_train_val_template.py"
# 1) Create a new copy of this file "set_local_experiment_train_val_template.py" in your local environment and rename it into "set_local_experiment_train_val.py"
# 2) Indicate all the variables according to your local environment and experiment
# 3) Use your own "set_local_experiment_train_val.py" file to run the code
# 4) Do not commit/push your own "set_local_experiment_train_val.py" file to the collective repository, it is not relevant for other people
# 5) The untracked file "set_local_experiment_train_val.py" is automatically copied to the output folder for the model weights for reproductibility
# ****************************************************************
from set_locals import experiment

def set_local_experiment_train_val():
    e = experiment.ExperimentTrainVal(
        # --- Path and Data
        path_noise_samples='???',               # Path to experimentally acquired noise sinograms in separate folders 'train' and 'val'
        path_signal_samples='???',              # Path to simulated optoacoustic signals in separate folders 'train' and 'val'
        path_test_samples='???',                # Path to invivo test sinograms (only one sinogram is required and printed in tensorboard after each epoch).
        save_path='???',                        # Output path for the saved training parameters, logs, and the trained weights
        experiment_name='foo',                  # Unique experiment name (used as folder names when saving the data)
        name_printed_signal_val='???',          # Name of the simulated sinogram from the val split (with extension) that will be displayed in tensorboard after each epoch.
        name_printed_signal_test='???',         # Name of a invivo test sinogram (with extension) that is printed in tensorboard after each epoch.
        clim_printed_sinograms=[-3, 3],         # Min. and max. value of the color bar for all printed sinograms in tensorboard
        gpu_index=0,                            # Index of the GPU that is used for training
        scaling_factor_for_signal_samples=100.0,    # Scaling factors for simulated optoacoustic to adjust to the values recorded invivo.
        random_scaling_for_signal_samples=True,     # False: Use always the same scaling factor; True: Draw scaling factors uniformly at random from (0,scaling_factor_for_signal_samples].
        divisor_for_data_normalization=100.0,       # Normalization factor for all input data.
        batch_size_train=1,                         # Batch size during training
        batch_size_val=1,                           # Batch size during validation
        loss_nickname='MAE',                        # Name of the training loss, see 'custom_loss.py' for all options.
        num_epochs=200,                             # Number of training epochs
        optimizer='sgd',                            # Name of the optimizer, see 'get_optimizer.py' for all options.
        learning_rate=0.001,                        # Learning rate
        momentum=0.99,  # for 'sgd'                 # Momentum when using the 'sgd' optimizer (otherwise the parameter is ignored)
        betas=(0.5, 0.999),                         # Beta values when using the 'adam' optimizer (otherwise the parameter is ignored)
        scheduler='step',                           # Name of the used scheduler, see 'get_scheduler.py' for all options.
        scheduler_step_size=1,                      # Step size if the scheduler 'step' is used (otherwise the parameter is ignored)
        scheduler_gamma=0.99,                       # Gamma value if the scheduler 'step' is used (otherwise the parameter is ignored)
        scheduler_num_epoch_linear_decrease=50,     # Number of epochs during which the learning rate is decrease to zero for scheduler 'linear' (otherwise the parameter is ignored).
        ngf_unet=64,                                # Number of U-net filters
        num_layers_unet=5,                          # Number of U-net layers
        use_bias=False,                             # Enable or disable bias terms in the U-Net
        include_timestps_channel=False,             # Include the recording time for all values as an additional input channel to the network (experimental feature).
        gain_for_normal_weight_init=0.02            # Gain value for normal weight initialization (ignored if zero or negative value is provided).
    )

    # --- Print all attributes in the console
    attrs = vars(e)
    print('\n'.join("%s: %s" % item for item in attrs.items()))
    print('----------------------------------------------------------------')

    # --- Return populated object from Experiment class
    return e
