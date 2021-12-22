function preprocessed_data = crop_window_and_filter_msot_data(data_raw, num_cropped_samples_sinogram_start, num_cropped_samples_sinogram_end, preproc_window_length_start, preproc_window_length_end, preproc_window_butter_degree, preproc_min_filt_freq, preproc_max_filt_freq, dac_frequency, apply_filtering, apply_windowing)
    
    preprocessed_data = crop_first_n_signals(data_raw,  num_cropped_samples_sinogram_start);
    preprocessed_data = preprocessed_data(:,:, 1: ( end-num_cropped_samples_sinogram_end), :);
    if apply_windowing
        preprocessed_data = apply_butterworth_window_to_sinogram(preprocessed_data, preproc_window_butter_degree, preproc_window_length_start, size(preprocessed_data,1)-preproc_window_length_end);
    end
    if apply_filtering
        preprocessed_data = filter_butter_zero_phase(preprocessed_data, dac_frequency, [preproc_min_filt_freq, preproc_max_filt_freq], true);
    end
    
end

