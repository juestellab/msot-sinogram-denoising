% ****************************************************************
% *** HOWTO
% ****************************************************************
% 0) Do not modify this template file "set_parameters_for_reconstruction_template.py"
% 1) Create a new copy of this file "set_parameters_for_reconstruction_template.py" in your local environment and rename it into "set_parameters_for_reconstruction.py"
% 2) Indicate all the variables according to your local environment and experiment
% 3) Use your own "set_parameters_for_reconstruction.py" file to run the code
% 4) Do not commit/push your own "set_parameters_for_reconstruction.py" file to the collective repository, it is not relevant for other people
% 5) The untracked file "set_parameters_for_reconstruction.py" is automatically copied to the reconstruction output folder for reproductibility
% ****************************************************************
function [...
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
    lambda] = set_parameters_for_reconstruction()

    % Parameters for data loading
    % Note: The data loading routine is specific to the used scanner and  probably requires adjustments if a different scanner is used.
    path_to_rec_toolbox = '???';                % Path to model-based reconstruction toolbox.
    path_sinograms_folder = '';                 % Path to folder that contains noisy or denoised sinograms
    path_to_used_preprocessing_parameters = ''; % Path to saved preprocessing parameters for the sinograms. If empty, default value is used: <path_sinograms_folder>/../preprocessing_params.mat
    path_to_csv = '???';                        % Path to csv file that specifies the sinograms that are reconstructed and the speed of sound value that should be used.
    addpath(genpath(path_to_rec_toolbox));      % required here for 'read_studies_scans_selmats_from_csv'
    [~, ~, ~, speed_of_sounds_tissue, ~, names] = read_studies_scans_selmats_from_csv(path_to_csv, 1); % Load speed of sounds and names of the sinograms that should be reconstructed.
    
    % Parameters from the model-based reconstruction toolbox for non-negative model-bared reconstruction
    use_eir = true;                             % Include the electrical impulse response in the model.
    use_indiv_eir = true;                       % Include the individual electrical impulse response in the model.
    use_sir = true;                             % Include the spatial impulse response in the model.
    use_single_speed_of_sound = false;          % Use a single SoS model instead of a duals SoS model in the model.
    field_of_view = [-0.02 0.02 -0.02 0.02];    % Field of view for the model.
    number_of_grid_points_fov = [401 401];      % Image size for the model.
    num_iterations = 50;                        % Number of iterations.           
    lambda = 1e-2;                              % Regularization value 
end

