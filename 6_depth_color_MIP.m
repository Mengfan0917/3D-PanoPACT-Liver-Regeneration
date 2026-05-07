% =========================================================================
% 3D-PanoPACT Depth Color-Encoded MIP Generation
% Maps vessel depth (Z-axis) to HSV color and exports depth-encoded
% maximum intensity projection for anatomical localization.
% =========================================================================

clc; clear;

%% 1. Setup
addpath('');  % Path containing colorMIP.m
save_path = '';
if ~exist(save_path, 'dir'), mkdir(save_path); end

%% 2. Load enhanced vessel volume
load('vessel_enhanced_grayscale_pure_black.mat');

% Flip Z-axis to match anatomical depth orientation (ventral = shallow)
V = flip(double(V), 3);

%% 3. Depth color encoding
% colorMIP maps Z-slice index to HSV hue, producing a depth-colored MIP.
% Shallow structures appear red/magenta; deep structures appear blue/green.
pa_deep_xy = colorMIP(V(:,:,220:370));

%% 4. Visualization
figure(101);
imshow(pa_deep_xy, []);
axis image;
title('3D-PanoPACT Depth Color-Encoded MIP');

%% 5. Save results
imwrite(pa_deep_xy, fullfile(save_path, 'ColorMIP.png'));
save(fullfile(save_path, 'pa_deep_xy_result.mat'), 'pa_deep_xy');
fprintf('Depth color-encoded MIP saved.\n');
