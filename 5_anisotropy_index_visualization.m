% =========================================================================
% 3D Hepatic Vascular Anisotropy Index (AI) Computation and Visualization
% Computes vascular distribution diversity (VDD), morphological irregularity
% (MMI), and composite anisotropy index (AI) from enhanced PA volume.
% =========================================================================

%% 1. Load and preprocess volume
fprintf('Loading and preprocessing volume...\n');
data  = load('vessel_enhanced_grayscale_pure_black.mat');
V_raw = double(data.V(:,:,185:485));  % Crop to hepatic region of interest

V_raw = imgaussfilt3(V_raw, 1.0);     % Gaussian smoothing

% Binarization: 99.2nd percentile threshold + connected-component filtering
threshold   = prctile(V_raw(:), 99.2);
V_bin_full  = V_raw > threshold;
V_bin_clean = bwareaopen(V_bin_full, 1000);

% Downsample for computational efficiency
scale = 0.5;
fprintf('Downsampling by %.1f...\n', scale);
V_bin      = imresize3(double(V_bin_clean), scale) > 0.5;
V_raw_small = imresize3(V_raw, scale);
[H, W, D]  = size(V_bin);

%% 2. Vessel skeleton extraction and grayscale mapping
vessel_skeleton_bin = bwskel(V_bin);
vs_grayscale = double(vessel_skeleton_bin) .* V_raw_small;

%% 3. Vascular Distribution Diversity (VDD) via 3D entropy filtering
vdd_map = entropyfilt(double(V_bin), true(5,5,5));

%% 4. Morphological Irregularity (MMI) via structure tensor analysis
[Gx, Gy, Gz] = gradient(V_raw_small);
sigma = 1.5;

% Compute smoothed structure tensor components
Jxx = imgaussfilt3(Gx.^2,    sigma);  Jyy = imgaussfilt3(Gy.^2,    sigma);  Jzz = imgaussfilt3(Gz.^2,    sigma);
Jxy = imgaussfilt3(Gx.*Gy,   sigma);  Jxz = imgaussfilt3(Gx.*Gz,   sigma);  Jyz = imgaussfilt3(Gy.*Gz,   sigma);

idx = find(V_bin);
num_pts = length(idx);
mmi_values = zeros(num_pts, 1);
[ix, iy, iz] = ind2sub([H, W, D], idx);

% Parallel computation of per-voxel anisotropy from eigenvalue decomposition
if isempty(gcp('nocreate')), parpool; end
parfor k = 1:num_pts
    i = ix(k);  j = iy(k);  l = iz(k);
    M  = [Jxx(i,j,l), Jxy(i,j,l), Jxz(i,j,l);
          Jxy(i,j,l), Jyy(i,j,l), Jyz(i,j,l);
          Jxz(i,j,l), Jyz(i,j,l), Jzz(i,j,l)];
    ev = sort(eig(M), 'descend');
    mmi_values(k) = (ev(1) - ev(3)) / (sum(ev) + eps);
end
mmi_map       = zeros(H, W, D);
mmi_map(idx)  = mmi_values;

%% 5. Composite Anisotropy Index (AI)
AI_map = vdd_map .* mmi_map;

%% 6. Generate and export projection images
fprintf('Generating high-resolution projection images...\n');

% Maximum intensity projections (XY plane)
vs_gray_xy = squeeze(max(vs_grayscale, [], 3));
ai_xy      = squeeze(max(AI_map,       [], 3));

% Combined comparison figure (for preview)
hFig = figure('Color', 'w', 'Position', [100 100 1400 600]);
ax1  = subplot(1,2,1);
imagesc(vs_gray_xy); axis image off; colormap(ax1, gray); colorbar;
title('Vessel Skeleton (VS) Gray Projection', 'FontSize', 14, 'FontWeight', 'bold');

ax2 = subplot(1,2,2);
imagesc(ai_xy); axis image off; colormap(ax2, jet); colorbar;
clim([0 0.9]);   % Unified color scale across all AI figures
title('Anisotropy Index (AI) XY Projection', 'FontSize', 14, 'FontWeight', 'bold');

exportgraphics(hFig, 'Vessel_Analysis_Comparison.png', 'Resolution', 600);
fprintf('Comparison figure saved: Vessel_Analysis_Comparison.png\n');

% High-resolution skeleton (no colorbar, for manuscript layout)
figure('Visible', 'off');
imagesc(vs_gray_xy); axis image off; colormap(gray);
exportgraphics(gca, 'Figure_Skeleton_HighRes.png', 'Resolution', 600);

% High-resolution AI map (no colorbar, unified color scale)
figure('Visible', 'off');
imagesc(ai_xy); axis image off; colormap(jet);
clim([0 0.9]);
exportgraphics(gcf, 'Figure_AI_Index_HighRes.png', 'Resolution', 600);

fprintf('High-resolution figures saved. AI color scale unified to [0, 0.9].\n');

%% 7. Export binary skeleton MIP
fprintf('Generating binary skeleton projection...\n');

vs_bin_xy = squeeze(max(vessel_skeleton_bin, [], 3));

hFigBin = figure('Visible', 'off');
imagesc(vs_bin_xy); colormap(gray); clim([0 1]);
axis image off;
set(gca, 'LooseInset', [0,0,0,0]);
exportgraphics(gca, 'Figure_Skeleton_Binary_Pure.png', 'Resolution', 600);

fprintf('Binary skeleton projection saved: Figure_Skeleton_Binary_Pure.png\n');
