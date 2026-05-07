% =========================================================================
% 3D Hepatic Vascular Quantification
% Binarizes enhanced PA volume, extracts 3D skeleton, and computes
% quantitative vascular parameters: volume, length, diameter, branching.
% =========================================================================

%% 1. Load enhanced vessel volume
load('vessel_enhanced_grayscale_pure_black.mat');

% Binarize: V > 0 retains only enhanced, background-suppressed voxels
BW = V > 0;

%% 2. Physical resolution definition
vox_size   = 30 / 601;    % Voxel edge length (mm)
vox_volume = vox_size^3;  % Voxel volume (mm^3)

%% 3. 3D skeletonization (centerline extraction)
fprintf('Extracting 3D skeleton...\n');
skel = bwskel(BW);

%% 4. Quantitative vascular parameter extraction
% A. Total vessel volume (mm^3)
vessel_vol_mm3 = sum(BW(:)) * vox_volume;

% B. Total vessel length (mm)
total_length_mm = sum(skel(:)) * vox_size;

% C. Mean vessel diameter (mm)
% bwdist gives distance to nearest background voxel; value at skeleton = radius
D = bwdist(~BW);
avg_diameter_mm = 2 * mean(D(skel)) * vox_size;

% D. Branching point density
% Skeleton voxels with >3 neighbors in 3x3x3 kernel are branch points
kernel      = ones(3, 3, 3);
branch_map  = convn(double(skel), kernel, 'same') .* skel;
branch_points = sum(branch_map(:) > 3);

%% 5. Display and export results
fprintf('--- Quantitative Vascular Analysis ---\n');
fprintf('Total vessel volume (mm^3): %.4f\n', vessel_vol_mm3);
fprintf('Total vessel length (mm):   %.4f\n', total_length_mm);
fprintf('Mean vessel diameter (mm):  %.4f\n', avg_diameter_mm);
fprintf('Branching point count:      %d\n',   branch_points);

% Export to Excel
metrics_names  = {'Vessel_Total_Volume_mm3'; 'Total_Length_mm'; 'Average_Diameter_mm'; 'Branch_Points_Count'};
metrics_values = [vessel_vol_mm3; total_length_mm; avg_diameter_mm; branch_points];
result_table   = table(metrics_names, metrics_values, 'VariableNames', {'Metric', 'Value'});
writetable(result_table, 'Vessel_Analysis_Results.xlsx');

%% 6. 3D visualization
figure('Color', 'w');

% Vessel surface rendered as semi-transparent mesh
fv = isosurface(BW, 0.5);
patch(fv, 'FaceColor', [0.6 0 0], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
hold on;

title('3D Vascular Segmentation and Skeleton');
axis equal; grid on; view(3);
camlight headlight; lighting gouraud; material shiny;

ax = gca;
[d1, d2, d3] = size(BW);
xlim(ax, [0, d2]); ylim(ax, [0, d1]); zlim(ax, [0, d3]);
xlabel('X (pixels)'); ylabel('Y (pixels)'); zlabel('Z (pixels)');
set(ax, 'Box', 'on');

%% 7. Save results and figure
AnalysisData.Metrics.TotalVolume_mm3    = vessel_vol_mm3;
AnalysisData.Metrics.TotalLength_mm     = total_length_mm;
AnalysisData.Metrics.AverageDiameter_mm = avg_diameter_mm;
AnalysisData.Metrics.BranchPointsCount  = branch_points;
AnalysisData.Params.VoxSize   = vox_size;
AnalysisData.Params.Timestamp = datetime('now');

set(gcf, 'UserData', AnalysisData);
saveas(gcf, 'Vessel_Skeleton_Result_Final.fig');
fprintf('Analysis complete. Results saved to Excel and .fig file.\n');
