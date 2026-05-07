% =========================================================================
% 3D Hepatic Vascular Enhancement and MIP Export
% Loads segmented PA volume, applies Frangi-based vessel enhancement,
% background purification, and exports maximum intensity projection images.
% =========================================================================

%% 1. Load and preprocess segmented volume
load('seg_results.mat');
img = double(image_intercepted);

% Normalize to [0, 1]
img = (img - min(img(:))) / (max(img(:)) - min(img(:)));

% 3D median filter for denoising
img_denoised = medfilt3(img, [7 7 7]);

%% 2. Vessel enhancement (Frangi / Hessian-based)
V = fibermetric(img_denoised, 8, 'ObjectPolarity', 'bright');

% Step A: Remove isolated noise via connected-component filtering
fprintf('Removing isolated noise from enhanced volume...\n');
tmp_BW         = V > (graythresh(V) * 0.5);
tmp_BW_cleaned = bwareaopen(tmp_BW, 2000);
V(tmp_BW == 1 & tmp_BW_cleaned == 0) = 0;

% Step B: Global background suppression (ensures pure-black background in saved volume)
fprintf('Applying global background suppression...\n');
bg_threshold_3d = max(V(:)) * 0.15;  % 15% of peak intensity
V(V < bg_threshold_3d) = 0;

% Save background-suppressed enhanced grayscale volume
save('vessel_enhanced_grayscale_pure_black.mat', 'V', '-v7.3');
fprintf('Enhanced volume saved: vessel_enhanced_grayscale_pure_black.mat\n');

%% 3. Export MIP projections at 600 DPI
view_names = {'YZ_View', 'XZ_View', 'XY_View'};
hFig = figure('Visible', 'off');

for dim = 1:3
    mip_data  = squeeze(max(V, [], dim));   % Maximum intensity projection
    mip_norm  = mat2gray(mip_data);         % Linear normalization to [0, 1]
    mip_final = imrotate(mip_norm, 90);     % Rotate for standard anatomical orientation

    imshow(mip_final, 'Border', 'tight');
    img_name = sprintf('Vessel_MIP_Natural_%s_600DPI.png', view_names{dim});
    exportgraphics(gca, img_name, 'Resolution', 600);
    fprintf('MIP saved: %s\n', img_name);
end

close(hFig);
