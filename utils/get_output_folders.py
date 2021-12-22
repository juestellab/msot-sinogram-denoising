import os


# ----------------------------------------------------------------
def get_output_folders_for_train_val(save_path, experiment_name, create_missing_folders=False):
    # --- Root for all tensorboard logs and saved model weights
    tensorboard_base_path = os.path.join(save_path, 'tensorboard')
    model_weights_base_path = os.path.join(save_path, 'model_weights')

    # --- Folders for the current run
    tensorboard_output_dir = os.path.join(tensorboard_base_path, experiment_name)
    model_weights_output_dir = os.path.join(model_weights_base_path, experiment_name)

    # --- Create folders if not exist
    if create_missing_folders:
        create_folder_if_not_exist(save_path)
        create_folder_if_not_exist(tensorboard_base_path)
        create_folder_if_not_exist(tensorboard_output_dir)
        create_folder_if_not_exist(model_weights_base_path)
        create_folder_if_not_exist(model_weights_output_dir)

    # --- Verbose
    print("Tensorboard log folder: '" + tensorboard_output_dir + "'")
    print("Saved model weights folder: '" + model_weights_output_dir + "'")
    print('----------------------------------------------------------------')

    # --- Return
    return tensorboard_output_dir, model_weights_output_dir


# ----------------------------------------------------------------
def get_output_folder_for_infer(save_path, experiment_name):
    experiment_base_path = os.path.join(save_path, experiment_name)

    denoised_sinogrmas_path = os.path.join(experiment_base_path, 'sinograms_denoised')

    # --- Create folders if not exist
    create_folder_if_not_exist(experiment_base_path)
    create_folder_if_not_exist(denoised_sinogrmas_path)

    # --- Verbose
    print("Inference output folder: '" + experiment_base_path + "'")
    print('----------------------------------------------------------------')

    # --- Return
    return experiment_base_path, denoised_sinogrmas_path


# ----------------------------------------------------------------
def create_folder_if_not_exist(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
