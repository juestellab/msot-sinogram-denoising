class ExperimentTrainVal:

    def __init__(
            self,
            # Data parameters
            path_noise_samples,
            path_signal_samples,
            path_test_samples,
            save_path,
            experiment_name,
            name_printed_signal_val,
            name_printed_signal_test,
            clim_printed_sinograms,

            # DL parameter
            gpu_index,
            scaling_factor_for_signal_samples,
            random_scaling_for_signal_samples,
            divisor_for_data_normalization,
            batch_size_train,
            batch_size_val,
            loss_nickname,
            num_epochs,
            optimizer,
            learning_rate,
            momentum,
            betas,
            scheduler,
            scheduler_step_size,
            scheduler_gamma,
            scheduler_num_epoch_linear_decrease,
            ngf_unet,
            num_layers_unet,
            use_bias,
            include_timestps_channel,
            gain_for_normal_weight_init
    ):
        self.path_noise_samples = path_noise_samples
        self.path_signal_samples = path_signal_samples
        self.path_test_samples = path_test_samples
        self.save_path = save_path
        self.experiment_name = experiment_name
        self.name_printed_signal_val = name_printed_signal_val
        self.name_printed_signal_test = name_printed_signal_test
        self.clim_printed_sinograms = clim_printed_sinograms
        self.gpu_index = gpu_index
        self.scaling_factor_for_signal_samples = scaling_factor_for_signal_samples
        self.random_scaling_for_signal_samples = random_scaling_for_signal_samples
        self.divisor_for_data_normalization = divisor_for_data_normalization
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.loss_nickname = loss_nickname
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.betas = betas
        self.scheduler = scheduler
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_num_epoch_linear_decrease = scheduler_num_epoch_linear_decrease
        self.ngf_unet = ngf_unet
        self.num_layers_unet = num_layers_unet
        self.use_bias = use_bias
        self.include_timestps_channel = include_timestps_channel
        self.gain_for_normal_weight_init = gain_for_normal_weight_init


class ExperimentInfer:
    def __init__(
            self,
            # Data parameters
            path_noisy_input_sinograms,
            path_experiment_train_val_and_weights,
            save_path_infer,
            gpu_index_for_inference,
            regex_fullmatch_for_filenames):
        self.path_noisy_input_sinograms = path_noisy_input_sinograms
        self.path_experiment_train_val_and_weights = path_experiment_train_val_and_weights
        self.save_path_infer = save_path_infer
        self.gpu_index_for_inference = gpu_index_for_inference
        self.regex_fullmatch_for_filenames = regex_fullmatch_for_filenames
