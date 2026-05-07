% =========================================================================
% 3D Hepatic Vascular Segmentation
% Loads reconstructed PA volume, runs 3D cube interception and segmentation,
% and exports results and figures.%
% =========================================================================

%% 1. Initialization
clc;
clear;

%% 2. Add function search path
% Ensure Image_3D_cube_intercept.m and seg_manual.m are located here
addpath('');

%% 3. Load reconstructed PA volume
folder_path = '';
file_name   = '';
full_path   = fullfile(folder_path, file_name);

if exist(full_path, 'file')
    data_struct = load(full_path);
    var_names   = fieldnames(data_struct);
    image_data  = data_struct.(var_names{1}); % Read first variable in .mat file
    fprintf('Data loaded successfully. Size: %s\n', mat2str(size(image_data)));
else
    error('File not found: %s', full_path);
end

%% 4. Segmentation parameters
Is_Edge_smooth = 1;    % Edge smoothing: 1 = on, 0 = off
G_size         = 51;   % Gaussian kernel size (larger = smoother)
G_sigma        = 21;   % Gaussian standard deviation (larger = smoother)
CaxisNumber    = 0.5;  % Contrast scaling factor (0.1–1.0, adjust per image brightness)

%% 5. Run 3D segmentation
[image_intercepted, seg_volume] = Image_3D_cube_intercept( ...
    image_data, ...
    Is_Edge_smooth, ...
    G_size, ...
    G_sigma, ...
    CaxisNumber);

%% 6. Save segmentation results
save(fullfile(folder_path, 'seg_results.mat'), 'image_intercepted', 'seg_volume');
disp('Segmentation complete.');

%% 7. Export Figure 4 and Figure 5 as high-resolution PNG
figs_to_save = [4, 5];
for f = figs_to_save
    if ishandle(f)
        hFig = figure(f);
        set(hFig, 'Color', 'w');
        axis tight;
        drawnow;
        img_name = sprintf('Figure_%d_Result.png', f);
        exportgraphics(hFig, fullfile(folder_path, img_name), 'Resolution', 300);
        fprintf('Figure %d saved.\n', f);
    end
end
