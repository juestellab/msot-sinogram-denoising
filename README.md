# MSOT Sinogram Denoising
Deep-learning-based denoising of optoacoustic sinograms.

## Prerequisites
* For training and application of a denoising model, set up a python environment (v3.8 or higher) and install the following packages: `numpy`, `pytorch`, `tensorboard`, `medpy`, `natsort`.
* All scripts to generate data and reconstruct optoacoustic images require the model-based reconstruction toolbox available at https://github.com/juestellab/mb-rec-msot.
* In addition, the two functions for data loading must be implemented depending on 
Additonally, two-device specific data loading functions (`read_studies_scans_selmats_from_csv.m`,  `loadMSOTsignals.m`) must be implemented to 

## Denoising: Model training and inference
- ``train.py``: Main script to train a denoising model.
- ``set_locals/set_local_experiment_train_val[_template].py``: Routine used to define all the required parameters for model training.
- ``infer.py``: Main script to denoise optoacoustic sinograms with a trained model.
- ``set_locals/set_local_experiment_infer[_template].py``: Routine used to define all the required parameters for applying a trained model.

## Denoising: Data generation 
- ``matlab_dataset_generation/generate_dataset_of_noise_samples.m``: Script to generate a dataset of noise sinograms from scans of water for model training and validation.
- ``matlab_dataset_generation/generate_dataset_of_signal_samples.m``: Script to generate a dataset of synthetic optoacoustic sinogrmas based on general-feature images for model training and validation.
- ``matlab_dataset_generation/set_parameters_for_trainval_data_generation[_template].m``: Routine used to define all the required parameters for generating training and validation datasets.
- ``matlab_dataset_generation/generate_dataset_of_invivo_test_samples.m``: Script to generate a test dataset of invivo sinograms.
- ``matlab_dataset_generation/set_parameters_for_test_data_generation[_template].m``: Routine used to define all the required parameters for generating a test dataset.

Note: The two data loading functions in the above-mentioned scripts (`loadMSOTsignals.m` and `read_studies_scans_selmats_from_csv.m`) are device-specific and therefore not included in public version of the model-based reconstruction toolbox. Instead, these functions need be implemented separately depending on the structure and format of the data that should be used.

## Image reconstruction
- ``matlab_image_reconstruction/reconstruct_noisy_or_denoised_sinograms.m``: Script to reconstruct original or denoised singrams via model-based inversion.
- ``matlab_image_reconstruction/set_parameters_for_reconstruction[_template].m``: Routine used to define all the required reconstruction parameters.

## Citation
If you denoise optoacoustic sinograms with the source code in this repository, please cite this paper:
```
Dehner, C., Olefir, I., Basak Chowdhury, K., Jüstel, D., and Ntziachristos, V., “Deep learning based electrical noise removal enables high spectral optoacoustic contrast in deep tissue”, arXiv e-print, 2021, https://arxiv.org/abs/2102.12960.
```

## Contact
* Christoph Dehner (christoph.dehner@tum.de)