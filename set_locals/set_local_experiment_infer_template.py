# ****************************************************************
# *** HOWTO
# **************************************************************** 
# 0) Do not modify this template file "set_local_experiment_infer_template.py"
# 1) Create a new copy of this file "set_local_experiment_template.py" in your local environment and rename it into "set_local_experiment_infer.py"
# 2) Indicate all the variables according to your local environment and experiment
# 3) Use your own "set_local_experiment_infer.py" file to run the code
# 4) Do not commit/push your own "set_local_experiment_infer.py" file to the collective repository, it is not relevant for other people
# ****************************************************************
from set_locals import experiment


def set_local_experiment_infer():
    e = experiment.ExperimentInfer(
        path_noisy_input_sinograms='???',             # Path to invivo sinograms in a folder "test"
        path_experiment_train_val_and_weights='???',  # Path to saved weights, e.g. '[save path during training]/model_weights/[experiment name]'
        save_path_infer='???',                        # Output path for the denoised sinograms.
        gpu_index_for_inference=0,                    # Index of the GPU used for inference.
        regex_fullmatch_for_filenames='.*'            # Name filter for the processed invivo sinogram ("'.*'" to process all data).
    )

    # --- Print all attributes in the console
    attrs = vars(e)
    print('\n'.join("%s: %s" % item for item in attrs.items()))
    print('----------------------------------------------------------------')

    # --- Return populated object from Experiment class
    return e
