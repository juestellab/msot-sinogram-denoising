% Script to reconstruct noisy or denoised sinograms
[...
    path_to_rec_toolbox,...
    path_sinograms_folder,...
    path_to_used_preprocessing_parameters,...
    speed_of_sounds_tissue,...
    names,...
    use_eir,...
    use_indiv_eir,...
    use_sir,...
    use_single_speed_of_sound,...
    field_of_view,...
    number_of_grid_points_fov,...
    num_iterations,...
    lambda] = set_parameters_for_reconstruction();

%% Save parameters and initial model-based reconstruction toolbox
copyfile('./set_parameters_for_reconstruction.m', [path_sinograms_folder filesep '..' filesep 'set_parameters_for_reconstruction.m']);
run([path_to_rec_toolbox filesep 'startup_reconstruction.m']);

%% Get the saved preprocessing parameters of the input data
if isempty(path_to_used_preprocessing_parameters)
    path_to_used_preprocessing_parameters = [path_sinograms_folder filesep '..' filesep 'preprocessing_params.mat'];
end
load(path_to_used_preprocessing_parameters, ...
    'deviceId', ...
    'preproc_min_filt_freq', ...
    'preproc_max_filt_freq', ...
    'preproc_window_length_start', ...
    'preproc_window_length_end', ...
    'preproc_window_butter_degree', ...
    'num_cropped_samples_sinogram_start', ...
    'num_cropped_samples_sinogram_end');

%% Create output folder
save_dir_of_reconstructed_stacks = [path_sinograms_folder filesep '..' filesep 'reconstructed_stacks'];
if ~exist(save_dir_of_reconstructed_stacks, 'dir')
   mkdir(save_dir_of_reconstructed_stacks);
end

%% Start the reconstruction routine
sos_range = unique(speed_of_sounds_tissue);

for current_sos = sos_range'
  % Define model for reconstruction
    model = define_model_for_reconstruction(field_of_view, number_of_grid_points_fov, deviceId, use_eir, use_indiv_eir, use_sir, use_single_speed_of_sound, current_sos, num_cropped_samples_sinogram_start, preproc_min_filt_freq, preproc_max_filt_freq);
    
    data_indices_for_current_sos = find(speed_of_sounds_tissue==current_sos);
        
    % Iterate over data for current speed of sound and reconstruct
    for i_data = data_indices_for_current_sos'
        rec_save_path = [save_dir_of_reconstructed_stacks filesep names{i_data} '.nii'];
        if exist(rec_save_path, 'file')
            fprintf(['Skip "' names{i_data} '" because there is already a reconstruction saved.\n']);
            continue;
        end
        fprintf(['Reconstruct "' names{i_data} '"...\n']);

        dir_data_sinogram = dir([path_sinograms_folder filesep names{i_data} '*.nii']);
        dir_data_sinogram = sort({dir_data_sinogram.name});
        sinograms = zeros(2000, 256, length(dir_data_sinogram));
        for i_wavelength = 1:length(dir_data_sinogram)
            sinogram = niftiread([path_sinograms_folder filesep dir_data_sinogram{i_wavelength}]);
            sinograms(:,:,i_wavelength) = sinogram;
        end

        % Fill up any signals that were cropped at the sinogram end for the denoising
        if num_cropped_samples_sinogram_end > 0
            sinograms = cat(1, sinograms, zeros(num_cropped_samples_sinogram_end, size(sinograms, 2), size(sinograms, 3)));
        end
        
        % Use different function from the model-based reconstruction to change the regularization functional (e.g. 'rec_nn_with_L2_reg' for Tikhonov and Laplacian regularization)
        rec_stack = rec_nn_with_Shearlet_reg(model, double(sinograms), num_iterations, lambda);
        niftiwrite(rec_stack, rec_save_path);
    end

end








