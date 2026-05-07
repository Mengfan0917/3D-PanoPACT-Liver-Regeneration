% =========================================================================
% 3D-PanoPACT Reconstruction Code
% Photoacoustic/Ultrasound Dual-Modal 3D Reconstruction
% =========================================================================

%% Initialization
clc;
gpuDevice(1).reset();

folder_path = strcat('Original file\');
fixMatlabFilenames(folder_path);
str_name = dir(fullfile(folder_path, '*_0.mat'));
[datax, DAQ_time_point] = func_3D_PACT_Data_Time_Read(folder_path, str_name(141).name);

% Separate PA and US frames based on surface signal amplitude
frame1_val = max(sum(datax(:, 1:100, 1)));
frame2_val = max(sum(datax(:, 1:100, 2)));
offset = (frame1_val < frame2_val);
pa_idx = (1 + offset) : 2 : size(datax, 3);
us_idx = (2 - offset) : 2 : size(datax, 3);

data   = datax(:, :, pa_idx);
dataUS = permute(datax(:, :, us_idx), [2, 1, 3]);

Aline = mean(data, 3);
Aline = squeeze(mean(Aline, 1));
[~, DL1] = max(Aline(1:100));

detector = load('coordinate.txt');
detector(:,1) = detector(:,1) + 0.555;  % Coordinate calibration
detector(:,2) = detector(:,2) + 0.39;

[Nelemt, Nsample, Nframe] = size(data);

% Create output folders
folders = {'USresult', 'PAresult'};
for i = 1:length(folders)
    if ~exist(folders{i}, 'dir')
        mkdir(folders{i});
    end
end

%% System Parameters
% Reconstruction mode selector:
%   1: Single-speed CUDA | 2: Dual-speed CUDA | 3: Inner-speed iteration
%   4: Outer-speed iteration | 5: Single-speed iteration
%   6: Single-speed rotational compounding | 7: Dual-speed rotational compounding
%   8: Single-speed coherent-factor compounding | 9: Single/Dual-speed rotational-translational compounding
%   10: US single-speed | 11: US single-speed SOS sweep | 12: US rotational-translational compounding
%   13: PA fixed FOV, object translation compounding
reconstruct_mode = 9;

% Sound speed settings
T    = 22.8;         % Water temperature (°C)
V_M  = 1502.0;       % Single sound speed (m/s)
V_M_Range = 1496:0.5:1520;

% Ultrasound parameters
V_US = 1500;
US_FRAME_COMPOUND = 20;
Dynamic_Range = 20;  % dB
Is_Gating    = 1;    % 1: apply respiratory gating; 0: use all frames
Is_Denoising = 1;    % 1: apply sinogram denoising; 0: skip

VM_out       = 1502.5;           % Outer sound speed - water (m/s)
VM_out_Range = 1475:0.5:1499;
VM_in        = 1502.5;           % Inner sound speed - tissue (m/s)
VM_in_Range  = 1555:5:1699;

% Translational scan parameters
step_x        = 1;   % Step count in X (file direction)
step_y        = 1;   % Step count in Y (frame direction)
step_length_x = 6;   % X step size (mm)
step_length_y = 8;   % Y step size (mm)
Nframex_scan  = 7;
Ndata         = size(str_name, 1) / 4;
Nframey_scan  = 10;
GaussianMask_FWHM = 30; % Gaussian mask FWHM for sub-volume stitching (mm)

% Image reconstruction volume
x_size = 60;  y_size = 60;  z_size = 50;
resolution_factor = 10;
center_x = 0;  center_y = 0;  center_z = 0;

% Dual-speed ellipsoidal boundary parameters
Ellipse.a       = 15.0;   Ellipse.b       = 28.0;   Ellipse.c       = 13.5;
Ellipse.centerx = -2.7;   Ellipse.centery = -1.1;   Ellipse.centerz =  7.2;

% Sinogram denoising (removes out-of-bandwidth noise)
if Is_Denoising == 1
    data = denoise_sinogram(data);
end

predelay = -DL1;
pa_data  = -data;    % Negate for correct background polarity
fs = 40;             % Sampling frequency (MHz)
R  = 100;            % Spherical array radius (mm)

% Pixel grid computation
Npixel_x = x_size * resolution_factor + 1;
Npixel_y = y_size * resolution_factor + 1;
Npixel_z = z_size * resolution_factor + 1;
x_range = ((1:Npixel_x) - (Npixel_x+1)/2) * x_size/(Npixel_x-1) + center_x;
y_range = ((1:Npixel_y) - (Npixel_y+1)/2) * y_size/(Npixel_y-1) + center_y;
z_range = ((1:Npixel_z) - (Npixel_z+1)/2) * z_size/(Npixel_z-1) + center_z;
[X_img, Y_img, Z_img] = meshgrid(x_range, y_range, z_range);

% Affine transform: correct for stage-to-axis angle (48.38°)
theta_x = 0;  theta_y = 0;  theta_z = 48.38;
trans_x = 0;  trans_y = 0;  trans_z = 0;
rotate_x_mat = [1 0 0 0; 0 cosd(theta_x) -sind(theta_x) 0; 0 sind(theta_x) cosd(theta_x) 0; 0 0 0 1];
rotate_y_mat = [cosd(theta_y) 0 -sind(theta_y) 0; 0 1 0 0; sind(theta_y) 0 cosd(theta_y) 0; 0 0 0 1];
rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0; sind(theta_z) cosd(theta_z) 0 0; 0 0 1 0; 0 0 0 1];
trans_mat    = [1 0 0 trans_x; 0 1 0 trans_y; 0 0 1 trans_z; 0 0 0 1];
afine_mat    = trans_mat * rotate_x_mat * rotate_y_mat * rotate_z_mat;
detector_new = [detector, detector(:,1)*0+1] * afine_mat';

% Transfer sensor coordinates to GPU (z-axis inverted to match image frame)
x_sensor = gpuArray(single(detector_new(:,1)));
y_sensor = gpuArray(single(detector_new(:,2)));
z_sensor = gpuArray(single(-detector_new(:,3)));

% Transfer image grid to GPU
X_img      = gpuArray(single(X_img));
Y_img      = gpuArray(single(Y_img));
Z_img      = gpuArray(single(Z_img));
Points_img = cat(4, X_img, Y_img, Z_img);

%% Reconstruction Main Loop
tic
switch reconstruct_mode

    case 1 % Single-speed CUDA reconstruction
        for frame = 1
            pa_data_frame    = gpuArray(single(pa_data(:,:,frame)));
            Points_sensor_all = gpuArray(single([x_sensor, y_sensor, z_sensor]));
            [pa_img, total_angle_weight] = SingleSpeedReconstraction_mex( ...
                Points_sensor_all, Points_img, pa_data_frame, ...
                single(fs), single(predelay), single(V_M), single(R));
            disp(['frame: ', num2str(frame)]);

            pa_img1 = gather(pa_img);
            total_angle_weight = gather(total_angle_weight);
            pa_img2 = subplus(pa_img1) ./ total_angle_weight;

            imin = min(pa_img2,[],'all');  imax = max(pa_img2,[],'all');
            figure(2);
            subplot(131); imagesc(z_range, x_range, squeeze(max(pa_img2,[],1)),[imin,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('X'); xlabel('Z'); title('XZ proj');
            subplot(133); imagesc(z_range, y_range, squeeze(max(pa_img2,[],2)),[imin,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('Z'); title('YZ proj');
            subplot(132); imagesc(x_range, y_range, squeeze(max(pa_img2,[],3)),[imin,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('X'); title('XY proj');

            imwrite(mat2gray(squeeze(max(pa_img2(end:-1:1,:,:),[],1))), sprintf('PAresult/zx frame=%d,V_M=%.1f.png',frame,V_M));
            imwrite(mat2gray(squeeze(max(pa_img2(end:-1:1,:,:),[],2))), sprintf('PAresult/zy frame=%d,V_M=%.1f.png',frame,V_M));
            imwrite(mat2gray(squeeze(max(pa_img2(end:-1:1,:,:),[],3))), sprintf('PAresult/xy frame=%d,V_M=%.1f.png',frame,V_M));
        end

    case 2 % Dual-speed CUDA reconstruction
        for frame = 1
            pa_data_frame    = gpuArray(single(pa_data(:,:,frame)));
            Points_sensor_all = gpuArray(single([x_sensor, y_sensor, z_sensor]));
            [pa_img, total_angle_weight] = DualSpeedReconstraction_mex( ...
                [Ellipse.a,Ellipse.b,Ellipse.c,Ellipse.centerx,Ellipse.centery,Ellipse.centerz], ...
                Points_sensor_all, Points_img, pa_data_frame, ...
                single(fs), single(predelay), single(VM_out), single(VM_in), single(R));

            pa_img1 = gather(pa_img);
            total_angle_weight = gather(total_angle_weight);
            pa_img2 = subplus(pa_img1) ./ total_angle_weight;
            disp(['frame: ', num2str(frame)]);

            imin = min(pa_img2,[],'all');  imax = max(pa_img2,[],'all');
            figure(1);
            subplot(131); imagesc(z_range, x_range, squeeze(max(pa_img2,[],1)),[imin,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('X'); xlabel('Z'); title('XZ proj');
            subplot(133); imagesc(z_range, y_range, squeeze(max(pa_img2,[],2)),[imin,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('Z'); title('YZ proj');
            subplot(132); imagesc(x_range, y_range, squeeze(max(pa_img2,[],3)),[imin,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('X'); title('XY proj');

            imwrite(mat2gray(squeeze(max(pa_img2(end:-1:1,:,:),[],1))), sprintf('PAresult/zx frame=%d,VM_out=%.1f,VM_in=%.1f.png',frame,VM_out,VM_in));
            imwrite(mat2gray(squeeze(max(pa_img2(end:-1:1,:,:),[],2))), sprintf('PAresult/zy frame=%d,VM_out=%.1f,VM_in=%.1f.png',frame,VM_out,VM_in));
            imwrite(mat2gray(squeeze(max(pa_img2(end:-1:1,:,:),[],3))), sprintf('PAresult/xy frame=%d,VM_out=%.1f,VM_in=%.1f.png',frame,VM_out,VM_in));
        end

    case 3 % Inner sound speed sweep
        for VM_in = VM_in_Range
            for frame = 1
                pa_data_frame    = gpuArray(single(pa_data(:,:,frame)));
                Points_sensor_all = gpuArray(single([x_sensor, y_sensor, z_sensor]));
                tic
                [pa_img, total_angle_weight] = DualSpeedReconstraction_mex( ...
                    [Ellipse.a,Ellipse.b,Ellipse.c,Ellipse.centerx,Ellipse.centery,Ellipse.centerz], ...
                    Points_sensor_all, Points_img, pa_data_frame, ...
                    single(fs), single(predelay), single(VM_out), single(VM_in), single(R));
                toc
                pa_img1 = gather(pa_img);  total_angle_weight = gather(total_angle_weight);
                pa_img2 = subplus(pa_img1) ./ total_angle_weight;
                imax = max(pa_img2,[],'all');
                figure(3);
                subplot(131); imagesc(z_range,x_range,squeeze(max(pa_img2,[],1)),[1,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('X'); xlabel('Z'); title('XZ proj');
                subplot(133); imagesc(z_range,y_range,squeeze(max(pa_img2,[],2)),[1,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('Z'); title(['VM_in = ',num2str(VM_in)]);
                subplot(132); imagesc(x_range,y_range,squeeze(max(pa_img2,[],3)),[1,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('X'); title(['VM_out = ',num2str(VM_out)]);
                imwrite(mat2gray(squeeze(max(pa_img2(end:-1:1,:,:),[],1))), sprintf('PAresult/in zx VM_out=%.1f,VM_in=%.1f.png',VM_out,VM_in));
                imwrite(mat2gray(squeeze(max(pa_img2(end:-1:1,:,:),[],2))), sprintf('PAresult/in zy VM_out=%.1f,VM_in=%.1f.png',VM_out,VM_in));
                imwrite(mat2gray(squeeze(max(pa_img2(end:-1:1,:,:),[],3))), sprintf('PAresult/in xy VM_out=%.1f,VM_in=%.1f.png',VM_out,VM_in));
            end
        end

    case 4 % Outer sound speed sweep
        for VM_out = VM_out_Range
            for frame = 1
                pa_data_frame    = gpuArray(single(pa_data(:,:,frame)));
                Points_sensor_all = gpuArray(single([x_sensor, y_sensor, z_sensor]));
                tic
                [pa_img, total_angle_weight] = DualSpeedReconstraction_mex( ...
                    [Ellipse.a,Ellipse.b,Ellipse.c,Ellipse.centerx,Ellipse.centery,Ellipse.centerz], ...
                    Points_sensor_all, Points_img, pa_data_frame, ...
                    single(fs), single(predelay), single(VM_out), single(VM_in), single(R));
                toc
                pa_img1 = gather(pa_img);  total_angle_weight = gather(total_angle_weight);
                pa_img2 = subplus(pa_img1) ./ total_angle_weight;
                imax = max(pa_img2,[],'all');
                figure(3);
                subplot(131); imagesc(z_range,x_range,squeeze(max(pa_img2,[],1)),[1,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('X'); xlabel('Z'); title('XZ proj');
                subplot(133); imagesc(z_range,y_range,squeeze(max(pa_img2,[],2)),[1,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('Z'); title(['VM_in = ',num2str(VM_in)]);
                subplot(132); imagesc(x_range,y_range,squeeze(max(pa_img2,[],3)),[1,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('X'); title(['VM_out = ',num2str(VM_out)]);
                imwrite(mat2gray(squeeze(max(pa_img2(end:-1:1,:,:),[],1))), sprintf('PAresult/out zx VM_out=%.1f,VM_in=%.1f.png',VM_out,VM_in));
                imwrite(mat2gray(squeeze(max(pa_img2(end:-1:1,:,:),[],2))), sprintf('PAresult/out zy VM_out=%.1f,VM_in=%.1f.png',VM_out,VM_in));
                imwrite(mat2gray(squeeze(max(pa_img2(end:-1:1,:,:),[],3))), sprintf('PAresult/out xy VM_out=%.1f,VM_in=%.1f.png',VM_out,VM_in));
            end
        end

    case 5 % Single-speed sweep
        for V_M = V_M_Range
            for frame = 1
                tic
                pa_data_frame    = gpuArray(single(pa_data(:,:,frame)));
                Points_sensor_all = gpuArray(single([x_sensor, y_sensor, z_sensor]));
                [pa_img, total_angle_weight] = SingleSpeedReconstraction_mex( ...
                    Points_sensor_all, Points_img, pa_data_frame, ...
                    single(fs), single(predelay), single(V_M), single(R));
                toc
                pa_img1 = gather(pa_img);  total_angle_weight = gather(total_angle_weight);
                pa_img2 = subplus(pa_img1) ./ total_angle_weight;
                imax = max(pa_img2,[],'all');
                figure(4);
                subplot(131); imagesc(z_range,x_range,squeeze(max(pa_img2,[],1)),[1,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('X'); xlabel('Z'); title('XZ proj');
                subplot(133); imagesc(z_range,y_range,squeeze(max(pa_img2,[],2)),[1,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('Z'); title('YZ proj');
                subplot(132); imagesc(x_range,y_range,squeeze(max(pa_img2,[],3)),[1,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('X'); title(['VM = ',num2str(V_M)]);
                imwrite(mat2gray(squeeze(max(pa_img2(end:-1:1,:,:),[],1))), sprintf('PAresult/zx VM=%.1f.png',V_M));
                imwrite(mat2gray(squeeze(max(pa_img2(end:-1:1,:,:),[],2))), sprintf('PAresult/zy VM=%.1f.png',V_M));
                imwrite(mat2gray(squeeze(max(pa_img2(end:-1:1,:,:),[],3))), sprintf('PAresult/xy VM=%.1f.png',V_M));
            end
        end

    case 6 % Single-speed rotational compounding
        pa_total = zeros(size(Points_img(:,:,:,1)), 'single');

        % Correlation-based respiratory gating
        if Is_Gating == 1
            [T, D, F]    = size(pa_data(:,2501:3000,:));
            reshaped_data = reshape(pa_data(:,2501:3000,:), T*D, F);
            corr_mat     = corr(reshaped_data);
            corr_line    = mean(corr_mat, 1);
            corr_line    = corr_line / max(corr_line);
            corr_line(1:10) = 0;   % Discard pre-rotation frames
            static_frames   = 1:Nframe;
            Similarity_threshold = maxk(corr_line, 20); Similarity_threshold = Similarity_threshold(end);
            static_frames = static_frames(corr_line >= Similarity_threshold);
            figure(11); plot(corr_line,'b'); hold on;
            for isf = static_frames; plot(isf, corr_line(isf), '*r'); hold on; end; hold off;
        else
            static_frames = 1:Nframe;
        end

        delta_angle   = -0.800;   % Rotation step (°) at trigger speed 11000
        static_Nframe = size(static_frames, 2);
        firstframe_flag = 1;

        for frame = 1:static_Nframe
            tic
            theta_z   = (static_frames(frame)-1) * delta_angle;
            rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0; sind(theta_z) cosd(theta_z) 0 0; 0 0 1 0; 0 0 0 1];
            afine_mat    = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1] * rotate_z_mat;
            detector_corr = detector_new * afine_mat';

            pa_data_frame    = gpuArray(single(pa_data(:,:,static_frames(frame))));
            Points_sensor_all = gpuArray(single(detector_corr(:,1:3)));
            tic
            [pa_img, total_angle_weight] = SingleSpeedReconstraction_mex( ...
                Points_sensor_all, Points_img, pa_data_frame, ...
                single(fs), single(predelay), single(V_M), single(R));
            toc
            disp(['frame: ', num2str(frame)]);

            pa_img1 = gather(pa_img);  total_angle_weight = gather(total_angle_weight);
            pa_img2 = pa_img1 ./ total_angle_weight;
            if firstframe_flag; pa_ref = pa_img2; end
            pa_total = pa_total + pa_img2;
            firstframe_flag = 0;

            imax = max(pa_total,[],'all');
            figure(1);
            subplot(131); imagesc(z_range,x_range,squeeze(max(pa_total,[],1))); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('X'); xlabel('Z'); title('XZ proj');
            subplot(133); imagesc(z_range,y_range,squeeze(max(pa_total,[],2))); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('Z'); title('YZ proj');
            subplot(132); imagesc(x_range,y_range,squeeze(max(pa_total,[],3))); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('X'); title(frame);
            imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],1))), sprintf('PAresult/Single Speed Compounding zx frame=%d,V_M=%.1f.png',frame,V_M));
            imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],2))), sprintf('PAresult/Single Speed Compounding zy frame=%d,V_M=%.1f.png',frame,V_M));
            imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],3))), sprintf('PAresult/Single Speed Compounding xy frame=%d,V_M=%.1f.png',frame,V_M));
        end

    case 7 % Dual-speed rotational compounding
        pa_total = zeros(size(Points_img(:,:,:,1)));

        if Is_Gating == 1
            sub_data     = pa_data(:,2501:3000,:);
            [T, D, F]    = size(sub_data);
            reshaped_data = reshape(sub_data, T*D, F);
            corr_matrix  = corr(reshaped_data);
            corr_line    = mean(corr_matrix, 1) / max(mean(corr_matrix,1));
            static_frames = 1:Nframe;
            Similarity_threshold = maxk(corr_line, 20); Similarity_threshold = Similarity_threshold(end);
            static_frames = static_frames(corr_line >= Similarity_threshold);
            figure(11); plot(corr_line,'b'); hold on;
            for isf = static_frames; plot(isf, corr_line(isf), '*r'); hold on; end; hold off;
        else
            static_frames = 1:Nframe;
        end

        delta_angle   = -0.800;
        static_Nframe = size(static_frames, 2);
        firstframe_flag = 1;

        for frame = 1:static_Nframe
            theta_z = (static_frames(frame)-1) * delta_angle;
            rotate_x_mat = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
            rotate_y_mat = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
            rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0; sind(theta_z) cosd(theta_z) 0 0; 0 0 1 0; 0 0 0 1];
            trans_mat    = eye(4);
            afine_mat    = trans_mat * rotate_x_mat * rotate_y_mat * rotate_z_mat;
            detector_corr = detector_new * afine_mat';

            x_sensor = gpuArray(single(detector_corr(:,1)));
            y_sensor = gpuArray(single(detector_corr(:,2)));
            z_sensor = gpuArray(single(-detector_corr(:,3)));

            pa_data_frame    = gpuArray(single(pa_data(:,:,static_frames(frame))));
            Points_sensor_all = gpuArray(single([x_sensor,y_sensor,z_sensor]));
            [pa_img, total_angle_weight] = DualSpeedReconstraction_mex( ...
                [Ellipse.a,Ellipse.b,Ellipse.c,Ellipse.centerx,Ellipse.centery,Ellipse.centerz], ...
                Points_sensor_all, Points_img, pa_data_frame, ...
                single(fs), single(predelay), single(VM_out), single(VM_in), single(R));
            disp(['frame: ', num2str(frame)]);

            pa_img1 = gather(pa_img);  total_angle_weight = gather(total_angle_weight);
            pa_img2 = pa_img1 ./ total_angle_weight;
            pa_total = pa_total + pa_img2;
            firstframe_flag = 0;

            imax = max(pa_total,[],'all');
            figure(7);
            subplot(131); imagesc(z_range,x_range,squeeze(max(pa_total,[],1)),[0,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('X'); xlabel('Z'); title('XZ proj');
            subplot(133); imagesc(z_range,y_range,squeeze(max(pa_total,[],2)),[0,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('Z'); title('YZ proj');
            subplot(132); imagesc(x_range,y_range,squeeze(max(pa_total,[],3)),[0,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('X'); title('XY proj');
            drawEllipsoidOverlay(Ellipse);
            imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],1))), sprintf('PAresult/Dual Speed Compounding zx frame=%d,VM_out=%.1f,VM_in=%.1f.png',frame,VM_out,VM_in));
            imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],2))), sprintf('PAresult/Dual Speed Compounding zy frame=%d,VM_out=%.1f,VM_in=%.1f.png',frame,VM_out,VM_in));
            imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],3))), sprintf('PAresult/Dual Speed Compounding xy frame=%d,VM_out=%.1f,VM_in=%.1f.png',frame,VM_out,VM_in));
        end

    case 8 % Single-speed coherent-factor rotational compounding
        pa_total    = zeros(size(Points_img(:,:,:,1)));
        delta_angle = -10000*0.800/11000;
        theta_z     = delta_angle;
        rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0; sind(theta_z) cosd(theta_z) 0 0; 0 0 1 0; 0 0 0 1];
        afine_mat    = eye(4) * rotate_z_mat;

        for frame = 1:Nframe
            detector_new  = detector_new * afine_mat';
            x_sensor = gpuArray(single(detector_new(:,1)));
            y_sensor = gpuArray(single(detector_new(:,2)));
            z_sensor = gpuArray(single(-detector_new(:,3)));

            pa_data_frame    = gpuArray(single(pa_data(:,:,frame)));
            Points_sensor_all = gpuArray(single([x_sensor,y_sensor,z_sensor]));
            tic
            [pa_img, total_angle_weight, coherent_factor, ~] = SingleSpeedReconstraction_cof_mex( ...
                Points_sensor_all, Points_img, pa_data_frame, ...
                single(fs), single(predelay), single(V_M), single(R));
            toc
            disp(['frame: ', num2str(frame)]);

            pa_img1 = gather(pa_img);  total_angle_weight = gather(total_angle_weight);
            coherent_factor = gather(coherent_factor);
            pa_img2  = pa_img1 .* coherent_factor ./ total_angle_weight;
            pa_total = pa_total + pa_img2;

            imax = max(pa_total,[],'all');
            figure(1);
            subplot(131); imagesc(z_range,x_range,squeeze(max(pa_total,[],1)),[0,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('X'); xlabel('Z'); title('XZ proj');
            subplot(133); imagesc(z_range,y_range,squeeze(max(pa_total,[],2)),[0,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('Z'); title('YZ proj');
            subplot(132); imagesc(x_range,y_range,squeeze(max(pa_total,[],3)),[0,imax]); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('X'); title('XY proj');
            imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],1))), sprintf('zx frame=%d,V_M=%.1f.png',frame,V_M));
            imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],2))), sprintf('zy frame=%d,V_M=%.1f.png',frame,V_M));
            imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],3))), sprintf('xy frame=%d,V_M=%.1f.png',frame,V_M));
        end

    case 9 % Single/Dual-speed rotational-translational compounding (used for liver regeneration study)
        % Data acquisition protocol: capture first, then rotate.
        % Pre-rotation frames are automatically filtered via correlation gating.
        % Acquisition rate: 10 Hz.

        imgsize      = size(Points_img,1:3);
        pa_img_total = zeros(imgsize + [step_length_y*resolution_factor*(Nframey_scan - mod(Nframey_scan-1,step_y) - 1) ...
                                        step_length_x*resolution_factor*(Nframex_scan - mod(Nframex_scan-1,step_x) - 1) 0]);
        [totalsize_y, totalsize_x, totalsize_z] = size(pa_img_total);
        pa_count_total = zeros(totalsize_y, totalsize_x, totalsize_z, 'single');
        x_range_total  = -totalsize_x/resolution_factor/2 : totalsize_x/resolution_factor/2;
        y_range_total  = -totalsize_y/resolution_factor/2 : totalsize_y/resolution_factor/2;
        z_range_total  = -totalsize_z/resolution_factor/2 : totalsize_z/resolution_factor/2 + center_z;

        % Initialize display figures
        f9  = figure(9);
        subplot(131); h_img9_1 = imagesc(x_range, y_range, zeros(length(y_range), length(x_range))); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal'); ylabel('Y'); xlabel('X'); title('PA img XY'); set(gca,'tickdir','out');
        subplot(132); h_img9_2 = imagesc(x_range, y_range, zeros(length(y_range), length(x_range))); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal'); ylabel('Y'); xlabel('X'); title('Gaussian Mask XY'); set(gca,'tickdir','out');
        subplot(133); h_img9_3 = imagesc(x_range, y_range, zeros(length(y_range), length(x_range))); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal'); ylabel('Y'); xlabel('X'); title('PA img masked XY'); set(gca,'tickdir','out');

        f10 = figure(10);
        subplot(131); h_img10_1 = imagesc(z_range_total, x_range_total, zeros(length(x_range_total), length(z_range_total))); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('X'); xlabel('Z'); title('ZX proj');
        subplot(132); h_img10_2 = imagesc(x_range_total, y_range_total, zeros(length(y_range_total), length(x_range_total))); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('X'); title('XY proj');
        subplot(133); h_img10_3 = imagesc(z_range_total, y_range_total, zeros(length(y_range_total), length(z_range_total))); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('Z'); title('ZY proj');

        VM_out_start = VM_out;

        for xframe = 1:step_x:Nframex_scan
            for yframe = 1:step_y:Nframey_scan
                VM_in     = VM_out;
                frame_idx = 1 + ((xframe-1)*Nframey_scan + (yframe-1)) * 4;
                [datax, DAQ_time_point] = func_3D_PACT_Data_Time_Read(folder_path, str_name(frame_idx).name);

                frame1_val = max(sum(datax(:,1:100,1)));
                frame2_val = max(sum(datax(:,1:100,2)));
                offset = (frame1_val < frame2_val);
                pa_idx = (1+offset) : 2 : size(datax,3);
                pa_data = -datax(:,:,pa_idx);

                if Is_Denoising == 1
                    pa_data = denoise_sinogram(pa_data);
                end

                pa_data_frame = gpuArray(single(pa_data(:,:,1)));
                detector_new  = gpuArray(single([x_sensor, y_sensor, z_sensor, z_sensor*0+1]));
                pa_total = zeros(size(Points_img(:,:,:,1)), 'single');
                tic

                % Correlation-based gating to select quasi-static frames
                if Is_Gating == 1
                    [T, D, F]    = size(pa_data(:,2501:3000,:));
                    reshaped_data = reshape(pa_data(:,2501:3000,:), T*D, F);
                    corr_mat     = corr(reshaped_data);
                    corr_line    = mean(corr_mat, 1) / max(mean(corr_mat,1));
                    corr_line(1:10) = 0;
                    static_frames   = 1:Nframe;
                    Similarity_threshold = maxk(corr_line, 20); Similarity_threshold = Similarity_threshold(end);
                    static_frames = static_frames(corr_line >= Similarity_threshold);
                else
                    static_frames = 1:Nframe;
                end

                delta_angle   = -0.800;
                static_Nframe = size(static_frames, 2);
                firstframe_flag = 1;

                for frame = 1:static_Nframe
                    tic
                    theta_z = (static_frames(frame)-1) * delta_angle;
                    rotate_x_mat = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
                    rotate_y_mat = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
                    rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0; sind(theta_z) cosd(theta_z) 0 0; 0 0 1 0; 0 0 0 1];
                    trans_mat    = eye(4);
                    afine_mat    = trans_mat * rotate_x_mat * rotate_y_mat * rotate_z_mat;
                    detector_corr = detector_new * afine_mat';

                    pa_data_frame    = gpuArray(single(pa_data(:,:,static_frames(frame))));
                    Points_sensor_all = gpuArray(single(detector_corr(:,1:3)));
                    [pa_img, total_angle_weight] = DualSpeedReconstraction_mex( ...
                        [Ellipse.a,Ellipse.b,Ellipse.c,Ellipse.centerx,Ellipse.centery,Ellipse.centerz], ...
                        Points_sensor_all, Points_img, pa_data_frame, ...
                        single(fs), single(predelay), single(VM_out), single(VM_in), single(R));
                    disp(['xframe: ',num2str(xframe),' yframe: ',num2str(yframe),' frame: ',num2str(frame)]);

                    pa_img1 = gather(pa_img);  total_angle_weight = gather(total_angle_weight);
                    pa_img2 = pa_img1 ./ total_angle_weight;

                    % Apply Gaussian mask to reduce stitching artifacts
                    [GaussianMask, ~, ~] = generateGaussianMask({x_range, y_range}, 'Center', [center_x,center_y], 'Sigma', GaussianMask_FWHM*0.425);
                    GaussianMask = GaussianMask .* ones(1, 1, size(z_range,2));
                    pa_img3 = pa_img2 .* GaussianMask;
                    if firstframe_flag; pa_ref = pa_img3; end
                    pa_total = pa_total + subplus(pa_img3);
                    firstframe_flag = 0;
                    toc

                    set(h_img9_1, 'CData', squeeze(max(pa_img2,[],3)));
                    set(h_img9_2, 'CData', squeeze(max(GaussianMask,[],3)));
                    set(h_img9_3, 'CData', squeeze(max(pa_total,[],3)));
                    drawnow limitrate;

                    imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],1))), sprintf('zx xframe=%d, yframe=%d, frame=%d.png',xframe,yframe,frame));
                    imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],2))), sprintf('zy xframe=%d, yframe=%d, frame=%d.png',xframe,yframe,frame));
                    imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],3))), sprintf('xy xframe=%d, yframe=%d, frame=%d.png',xframe,yframe,frame));
                end % frame loop
                toc

                % Stitch sub-volume into global volume with Gaussian weighting
                SubWinSize_y = imgsize(1)-1;  SubWinSize_x = imgsize(2)-1;  SubWinSize_z = imgsize(3)-1;
                idx_y = totalsize_y - (yframe-1)*step_length_y*resolution_factor - SubWinSize_y : totalsize_y - (yframe-1)*step_length_y*resolution_factor;
                idx_x = (xframe-1)*step_length_x*resolution_factor + 1 : (xframe-1)*step_length_x*resolution_factor + 1 + SubWinSize_x;
                current_weight = GaussianMask * single(static_Nframe);
                pa_count_total(idx_y,idx_x,:) = pa_count_total(idx_y,idx_x,:) + current_weight;
                pa_img_total(idx_y,idx_x,:)   = pa_img_total(idx_y,idx_x,:)   + pa_total;
                pa_img_total_2 = pa_img_total ./ (pa_count_total + eps);

                % Rotation correction for stage-axis misalignment
                theta = 4;
                R_corr = [cosd(theta) -sind(theta) 0 0; sind(theta) cosd(theta) 0 0; 0 0 1 0; 0 0 0 1];
                pa_img_total_2 = imwarp(pa_img_total_2, affinetform3d(R_corr));

                imin = min(pa_img_total_2,[],'all');  imax = max(pa_img_total_2,[],'all');
                set(h_img10_1,'CData',squeeze(max(pa_img_total_2,[],1))); set(h_img10_1.Parent,'CLim',[imin,imax]);
                set(h_img10_2,'CData',squeeze(max(pa_img_total_2,[],3))); set(h_img10_2.Parent,'CLim',[imin,imax]);
                set(h_img10_3,'CData',squeeze(max(pa_img_total_2,[],2))); set(h_img10_3.Parent,'CLim',[imin,imax]);
                drawnow limitrate;

                imwrite(mat2gray(squeeze(max(pa_img_total_2(end:-1:1,:,:),[],1))), sprintf('step=%d zx xframe=%d, yframe=%d.png',step_x,xframe,yframe));
                imwrite(mat2gray(squeeze(max(pa_img_total_2(end:-1:1,:,:),[],2))), sprintf('step=%d zy xframe=%d, yframe=%d.png',step_x,xframe,yframe));
                imwrite(mat2gray(squeeze(max(pa_img_total_2(end:-1:1,:,:),[],3))), sprintf('step=%d xy xframe=%d, yframe=%d.png',step_x,xframe,yframe));
            end % yframe loop
        end % xframe loop

        figure();
        subplot(1,3,1); imagesc(squeeze(max(pa_img_total_2,[],1)));
        subplot(1,3,2); imagesc(squeeze(max(pa_img_total_2,[],3)));
        subplot(1,3,3); imagesc(squeeze(max(pa_img_total_2,[],2)));
        save('pa_img_total_step1.mat','pa_img_total_2','-v7.3');

    case 10 % Ultrasound rotational compounding
        [num_spl, num_rcv, num_rot] = size(dataUS);
        iq_ch = reshape(hilbert(reshape(dataUS, num_spl, [])), [num_spl, num_rcv, num_rot]);

        fs = 40e6;  fc = 3.5e6;  c = V_US;
        ang_hole = 16.5/180*pi;  f_xdc = 15e-3;  rad_xdc = 108.85e-3;
        del_tx = f_xdc/c;  rad_src = rad_xdc - f_xdc;
        x_src = -rad_src*sin(ang_hole);  y_src = 0;  z_src = rad_src*cos(ang_hole);
        src = [x_src, y_src, -z_src];

        rcv = detector * 1e-3;  rcv(:,3) = -rcv(:,3);
        no_rcv = 1:1024;  t0 = 100e-6;
        no_pos = 1:100;  a_rot = 0:-0.8:99*-0.811;

        x_1d = x_range*1e-3;  y_1d = y_range*1e-3;  z_1d = z_range*1e-3;
        num_xp = numel(x_1d);  num_yp = numel(y_1d);  num_zp = numel(z_1d);

        beta = 10/180*pi;  fd = 0;
        das_params = [t0, fs, fd, c, beta, del_tx];

        Rx = [1 0 0; 0 cosd(theta_x) -sind(theta_x); 0 sind(theta_x) cosd(theta_x)];
        Ry = [cosd(theta_y) 0 -sind(theta_y); 0 1 0; sind(theta_y) 0 cosd(theta_y)];
        Rz = [cosd(theta_z) -sind(theta_z) 0; sind(theta_z) cosd(theta_z) 0; 0 0 1];
        R_corr = Rz * Ry * Rx;

        if Is_Gating == 1
            [T, D, F] = size(dataUS);
            reshaped_data = reshape(dataUS, T*D, F);
            corr_matrix  = corr(reshaped_data);
            corr_line    = mean(corr_matrix, 1) / max(mean(corr_matrix,1));
            static_frames = 1:num_rot;
            Similarity_threshold = maxk(corr_line, US_FRAME_COMPOUND); Similarity_threshold = Similarity_threshold(end);
            static_frames = static_frames(corr_line >= Similarity_threshold);
        else
            static_frames = 1:US_FRAME_COMPOUND;
        end

        iq_im_sum = zeros(num_xp, num_yp, num_zp);

        for ii = 1:num_rot
            if ~ismember(ii, static_frames); continue; end
            tic;
            no_pos_i = no_pos(ii);
            R_i   = rotz(a_rot(no_pos_i));
            src_i = (R_i * src')';  rcv_i = (R_i * rcv(no_rcv,:)')';
            src_i = (R_corr * src_i')';  rcv_i = (R_corr * rcv_i')';
            ori_rcv_i = ([0,0,0] - rcv_i) ./ vecnorm([0,0,0]-rcv_i, 2, 2);

            iq_im_pos_i = mex_das_gpu(iq_ch(:,no_rcv,no_pos_i), x_1d, y_1d, z_1d, rcv_i, src_i, ori_rcv_i, das_params);
            iq_im_sum   = iq_im_sum + iq_im_pos_i;

            dr = Dynamic_Range;
            bm_im = abs(iq_im_sum) / max(abs(iq_im_sum),[],'all');
            bm    = 20*log10(bm_im);  bm(bm<-dr) = -dr;  bm_total = bm + dr;

            imwrite(mat2gray(squeeze(max(bm_total,[],2))), sprintf('USresult/US Compounding zx frame=%d,sos=%.1f.png',ii,c));
            imwrite(mat2gray(squeeze(max(bm_total,[],1))), sprintf('USresult/US Compounding zy frame=%d,sos=%.1f.png',ii,c));
            imwrite(mat2gray(squeeze(max(bm_total,[],3))), sprintf('USresult/US Compounding xy frame=%d,sos=%.1f.png',ii,c));
            fprintf('CUDA 3D Recon for Pos %d finished. %.4f sec used.\n', ii, toc);
        end

        bm_im = abs(iq_im_sum) / max(abs(iq_im_sum),[],'all');
        bm    = 20*log10(bm_im);  bm(bm<-30) = nan;
        figure();
        subplot(131); imagesc(z_range*1e-3, x_range*1e-3, squeeze(max(bm,[],2))); axis equal tight; colormap gray; colorbar; set(gca,'tickdir','out'); ylabel('X'); xlabel('Z'); title('XZ proj');
        subplot(133); imagesc(z_range*1e-3, y_range*1e-3, squeeze(max(bm,[],1))); axis equal tight; colormap gray; colorbar; set(gca,'tickdir','out'); ylabel('Y'); xlabel('Z'); title('YZ proj');
        subplot(132); imagesc(x_range*1e-3, y_range*1e-3, squeeze(max(bm,[],3))'); axis equal tight; colormap gray; colorbar; ylabel('Y'); xlabel('X'); title('XY proj'); set(gca,'tickdir','out');

    case 11 % Ultrasound sound speed sweep
        [num_spl, num_rcv, num_rot] = size(dataUS);
        iq_ch = reshape(hilbert(reshape(dataUS, num_spl, [])), [num_spl, num_rcv, num_rot]);
        fs = 40e6;  fc = 3.5e6;
        ang_hole = 16.5/180*pi;  f_xdc = 15e-3;  rad_xdc = 108.85e-3;
        rad_src = rad_xdc - f_xdc;
        src = [-rad_src*sin(ang_hole), 0, rad_src*cos(ang_hole)];
        rcv = detector*1e-3;  rcv(:,3) = -rcv(:,3);
        no_rcv = 1:1024;  t0 = 100e-6;
        no_pos = 1:100;  a_rot = 0:-0.8:99*-0.811;
        x_1d = x_range*1e-3;  y_1d = y_range*1e-3;  z_1d = z_range*1e-3;
        num_xp = numel(x_1d);  num_yp = numel(y_1d);  num_zp = numel(z_1d);
        beta = 10/180*pi;  fd = 0;
        Rx = [1 0 0; 0 cosd(theta_x) -sind(theta_x); 0 sind(theta_x) cosd(theta_x)];
        Ry = [cosd(theta_y) 0 -sind(theta_y); 0 1 0; sind(theta_y) 0 cosd(theta_y)];
        Rz = [cosd(theta_z) -sind(theta_z) 0; sind(theta_z) cosd(theta_z) 0; 0 0 1];
        R_corr = Rz * Ry * Rx;

        for sos = V_M_Range
            c = sos;  del_tx = f_xdc/c;
            das_params = [t0, fs, fd, c, beta, del_tx];
            iq_im_sum  = zeros(num_xp, num_yp, num_zp);
            for ii = 1
                tic;
                R_i   = rotz(a_rot(no_pos(ii)));
                src_i = (R_i * src')';  rcv_i = (R_i * rcv(no_rcv,:)')';
                src_i = (R_corr * src_i')';  rcv_i = (R_corr * rcv_i')';
                ori_rcv_i = ([0,0,0] - rcv_i) ./ vecnorm([0,0,0]-rcv_i, 2, 2);
                iq_im_pos_i = mex_das_gpu(iq_ch(:,no_rcv,no_pos(ii)), x_1d, y_1d, z_1d, rcv_i, src_i, ori_rcv_i, das_params);
                iq_im_sum   = iq_im_sum + iq_im_pos_i;
                fprintf('CUDA 3D Recon for Pos %d finished. %.4f sec used.\n', ii, toc);
            end
            dr = Dynamic_Range;
            bm_im = abs(iq_im_sum) / max(abs(iq_im_sum),[],'all');
            bm    = 20*log10(bm_im);  bm(bm<-dr) = -dr;  bm_total = bm + dr;
            imwrite(mat2gray(squeeze(max(bm_total,[],2))), sprintf('USresult/US SOS LOOP zx sos=%.1f.png',c));
            imwrite(mat2gray(squeeze(max(bm_total,[],1))), sprintf('USresult/US SOS LOOP zy sos=%.1f.png',c));
            imwrite(mat2gray(squeeze(max(bm_total,[],3))'), sprintf('USresult/US SOS LOOP xy sos=%.1f.png',c));
            figure(3);
            subplot(131); imagesc(z_range*1e-3, x_range*1e-3, squeeze(max(bm_total,[],2))); axis equal tight; colormap gray; colorbar; set(gca,'tickdir','out'); ylabel('X'); xlabel('Z'); title('XZ proj');
            subplot(133); imagesc(z_range*1e-3, y_range*1e-3, squeeze(max(bm_total,[],1))); axis equal tight; colormap gray; colorbar; set(gca,'tickdir','out'); ylabel('Y'); xlabel('Z'); title('YZ proj');
            subplot(132); imagesc(x_range*1e-3, y_range*1e-3, squeeze(max(bm_total,[],3))'); axis equal tight; colormap gray; colorbar; ylabel('Y'); xlabel('X'); title('XY proj'); set(gca,'tickdir','out');
        end

    case 12 % Ultrasound translational scan compounding
        scan_num = 70;
        fs = 40e6;  fc = 3.5e6;  c = V_US;
        ang_hole = 16.5/180*pi;  f_xdc = 15e-3;  rad_xdc = 108.85e-3;
        del_tx = f_xdc/c;  rad_src = rad_xdc - f_xdc;
        src = [-rad_src*sin(ang_hole), 0, rad_src*cos(ang_hole)];
        rcv = detector*1e-3;  rcv(:,3) = -rcv(:,3);
        no_rcv = 1:1024;  t0 = 100e-6;
        no_pos = 1:100;  a_rot = 0:-0.8:99*-0.811;
        x_1d = x_range*1e-3;  y_1d = y_range*1e-3;  z_1d = z_range*1e-3;
        num_xp = numel(x_1d);  num_yp = numel(y_1d);  num_zp = numel(z_1d);
        beta = 10/180*pi;  fd = 0;
        das_params = [t0, fs, fd, c, beta, del_tx];
        Rx = [1 0 0; 0 cosd(theta_x) -sind(theta_x); 0 sind(theta_x) cosd(theta_x)];
        Ry = [cosd(theta_y) 0 -sind(theta_y); 0 1 0; sind(theta_y) 0 cosd(theta_y)];
        Rz = [cosd(theta_z) -sind(theta_z) 0; sind(theta_z) cosd(theta_z) 0; 0 0 1];
        R_corr = Rz * Ry * Rx;

        iq_scan_all = zeros(num_xp, num_yp, num_zp, scan_num);
        scan = 1;

        for i = 3:4:280
            [datax, ~] = func_3D_PACT_Data_Time_Read(folder_now, str_name(i).name);
            frame1_val = max(sum(datax(:,1:100,1)));
            frame2_val = max(sum(datax(:,1:100,2)));
            offset = (frame1_val < frame2_val);
            us_idx = (2-offset):2:size(datax,3);
            data   = permute(datax(:,:,us_idx), [2,1,3]);
            [num_spl, num_rcv, num_rot] = size(data);
            iq_ch  = reshape(hilbert(reshape(data, num_spl, [])), [num_spl, num_rcv, num_rot]);
            tic

            reshaped_data = reshape(data, size(data,1)*size(data,2), size(data,3));
            corr_matrix   = corr(reshaped_data);
            corr_line     = mean(corr_matrix,1) / max(mean(corr_matrix,1));
            static_frames = 1:num_pos;
            Similarity_threshold = maxk(corr_line, US_FRAME_COMPOUND); Similarity_threshold = Similarity_threshold(end);
            static_frames = static_frames(corr_line >= Similarity_threshold);

            iq_im_sum = zeros(num_xp, num_yp, num_zp);
            for ii = 1:num_pos
                if ~ismember(ii, static_frames); continue; end
                tic;
                R_i   = rotz(a_rot(no_pos(ii)));
                src_i = (R_i * src')';  rcv_i = (R_i * rcv(no_rcv,:)')';
                src_i = (R_corr * src_i')';  rcv_i = (R_corr * rcv_i')';
                ori_rcv_i = ([0,0,0]-rcv_i) ./ vecnorm([0,0,0]-rcv_i, 2, 2);
                iq_im_pos_i = mex_das_gpu(iq_ch(:,no_rcv,no_pos(ii)), x_1d, y_1d, z_1d, rcv_i, src_i, ori_rcv_i, das_params);
                iq_im_sum   = iq_im_sum + iq_im_pos_i;
                fprintf('CUDA 3D Recon for Pos %d finished. %.4f sec used.\n', ii, toc);
            end
            iq_scan_all(:,:,:,scan) = iq_im_sum;
            disp(scan);  scan = scan + 1;
        end

        bm_Frame = zeros(num_xp, num_yp, num_zp, scan_num);
        dr = Dynamic_Range;
        for frame = 1:scan_num
            bm_im = abs(squeeze(iq_scan_all(:,:,:,frame))) / max(abs(squeeze(iq_scan_all(:,:,:,frame))),[],'all');
            bm    = 20*log10(bm_im);  bm(bm<-dr) = -dr;
            bm_Frame(:,:,:,frame) = bm + dr;
        end

        % Global volume stitching with Gaussian weighting
        dp = 1e-3 / resolution_factor;
        x_sp = [x_range(1), x_range(end)]*1e-3;  y_sp = [y_range(1), y_range(end)]*1e-3;
        Nx_global = round(((x_sp(2)-x_sp(1)) + (Nframex_scan-1)*step_length_x*1e-3) / dp) + 1;
        Ny_global = round(((y_sp(2)-y_sp(1)) + (Nframey_scan-1)*step_length_y*1e-3) / dp) + 1;
        Nz_global = num_zp;
        delta_x   = step_length_x*1e-3/dp;  delta_y = step_length_y*1e-3/dp;
        sigma_x = num_xp/4;  sigma_y = num_yp/4;
        [X_grid, Y_grid] = meshgrid(1:num_xp, 1:num_yp);
        G_2D = exp(-((X_grid-num_xp/2).^2/(2*sigma_x^2) + (Y_grid-num_yp/2).^2/(2*sigma_y^2)));
        GaussianMask  = repmat(single(G_2D), [1,1,num_zp]);
        Image_Global  = zeros(Nx_global, Ny_global, Nz_global, 'single');
        Weight_Global = zeros(Nx_global, Ny_global, Nz_global, 'single');

        for i = 1:Nframex_scan
            for j = 1:Nframey_scan
                x_index = (i-1)*delta_x;  y_index = (j-1)*delta_y;
                Image_Global (1+x_index:num_xp+x_index, Ny_global-num_yp+1-y_index:Ny_global-y_index, :) = ...
                    Image_Global(1+x_index:num_xp+x_index, Ny_global-num_yp+1-y_index:Ny_global-y_index, :) + bm_Frame(:,:,:,(i-1)*Nframey_scan+j).*permute(GaussianMask,[2,1,3]);
                Weight_Global(1+x_index:num_xp+x_index, Ny_global-num_yp+1-y_index:Ny_global-y_index, :) = ...
                    Weight_Global(1+x_index:num_xp+x_index, Ny_global-num_yp+1-y_index:Ny_global-y_index, :) + permute(GaussianMask,[2,1,3]);
            end
        end
        Weight_Global(Weight_Global==0) = 1;
        Image_Stitched = Image_Global ./ Weight_Global;

        figure(2);
        subplot(131); imagesc(squeeze(max(Image_Stitched,[],2))); axis equal tight; colormap gray; colorbar; set(gca,'tickdir','out'); ylabel('X'); xlabel('Z'); title('XZ proj');
        subplot(132); imagesc(squeeze(max(Image_Stitched,[],3))'); axis equal tight; colormap gray; colorbar; ylabel('Y'); xlabel('X'); title('XY proj'); set(gca,'tickdir','out');
        subplot(133); imagesc(squeeze(max(Image_Stitched,[],1))); axis equal tight; colormap gray; colorbar; set(gca,'tickdir','out'); ylabel('Y'); xlabel('Z'); title('YZ proj');

    case 13 % PA fixed FOV, object translational compounding
        % Sensor array and reconstruction FOV remain fixed.
        % Each sub-volume is placed at its physical translation offset in the global volume.
        imgsize      = size(Points_img,1:3);
        pa_img_total = zeros(imgsize + [step_length_y*resolution_factor*(Nframey_scan-mod(Nframey_scan-1,step_y)-1) ...
                                        step_length_x*resolution_factor*(Nframex_scan-mod(Nframex_scan-1,step_x)-1) 0]);
        [totalsize_y, totalsize_x, totalsize_z] = size(pa_img_total);
        pa_count_total = zeros(totalsize_y, totalsize_x, totalsize_z, 'single');
        x_range_total  = -totalsize_x/resolution_factor/2 : totalsize_x/resolution_factor/2;
        y_range_total  = -totalsize_y/resolution_factor/2 : totalsize_y/resolution_factor/2;
        z_range_total  = -totalsize_z/resolution_factor/2 : totalsize_z/resolution_factor/2 + center_z;

        f9  = figure(9);
        subplot(131); h_img9_1 = imagesc(x_range, y_range, zeros(length(y_range),length(x_range))); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal'); ylabel('Y'); xlabel('X'); title('PA img XY'); set(gca,'tickdir','out');
        subplot(132); h_img9_2 = imagesc(x_range, y_range, zeros(length(y_range),length(x_range))); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal'); ylabel('Y'); xlabel('X'); title('Gaussian Mask XY'); set(gca,'tickdir','out');
        subplot(133); h_img9_3 = imagesc(x_range, y_range, zeros(length(y_range),length(x_range))); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal'); ylabel('Y'); xlabel('X'); title('PA img masked XY'); set(gca,'tickdir','out');

        f10 = figure(10);
        subplot(131); h_img10_1 = imagesc(z_range_total,x_range_total,zeros(length(x_range_total),length(z_range_total))); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('X'); xlabel('Z'); title('ZX proj');
        subplot(132); h_img10_2 = imagesc(x_range_total,y_range_total,zeros(length(y_range_total),length(x_range_total))); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('X'); title('XY proj');
        subplot(133); h_img10_3 = imagesc(z_range_total,y_range_total,zeros(length(y_range_total),length(z_range_total))); axis equal tight; colormap gray; colorbar; set(gca,'YDir','normal','tickdir','out'); ylabel('Y'); xlabel('Z'); title('ZY proj');

        VM_out_start = VM_out;

        for xframe = 1:step_x:Nframex_scan
            for yframe = 1:step_y:Nframey_scan
                VM_in     = VM_out;
                frame_idx = 1 + ((xframe-1)*Nframey_scan + (yframe-1)) * 4;
                [datax, ~] = func_3D_PACT_Data_Time_Read(folder_path, str_name(frame_idx).name);

                frame1_val = max(sum(datax(:,1:100,1)));
                frame2_val = max(sum(datax(:,1:100,2)));
                offset = (frame1_val < frame2_val);
                pa_idx = (1+offset):2:size(datax,3);
                pa_data = -datax(:,:,pa_idx);
                if Is_Denoising == 1; pa_data = denoise_sinogram(pa_data); end

                x_sensor_new = x_sensor;  y_sensor_new = y_sensor;
                detector_new = gpuArray(single([x_sensor_new, y_sensor_new, z_sensor, z_sensor*0+1]));

                % Update FOV center to current translation position
                center_x = -1.75 - 40 + (xframe-1)*step_x*step_length_x;
                center_y =  3.35 - 40 + (yframe-1)*step_y*step_length_y;
                center_z = 0;
                x_range  = ((1:Npixel_x)-(Npixel_x+1)/2)*x_size/(Npixel_x-1) + center_x;
                y_range  = ((1:Npixel_y)-(Npixel_y+1)/2)*y_size/(Npixel_y-1) + center_y;
                z_range  = ((1:Npixel_z)-(Npixel_z+1)/2)*z_size/(Npixel_z-1) + center_z;
                [X_img, Y_img, Z_img] = meshgrid(x_range, y_range, z_range);
                X_img = gpuArray(single(X_img));  Y_img = gpuArray(single(Y_img));  Z_img = gpuArray(single(Z_img));
                Points_img = cat(4, X_img, Y_img, Z_img);
                pa_total = zeros(size(Points_img(:,:,:,1)), 'single');

                % Correlation-based gating
                corr_mat = zeros(Nframe, Nframe);
                tic
                for frx = 1:Nframe
                    for fry = frx:Nframe
                        tmp = corrcoef(pa_data(:,2501:3000,frx), pa_data(:,2501:3000,fry));
                        corr_mat(frx,fry) = tmp(1,2);
                    end
                end
                corr_mat  = corr_mat + corr_mat';
                corr_line = mean(corr_mat, 1) / max(mean(corr_mat,1));
                corr_line(1:9) = 0;
                static_frames = 1:Nframe;
                Similarity_threshold = maxk(corr_line, 20); Similarity_threshold = Similarity_threshold(end);
                static_frames = static_frames(corr_line >= Similarity_threshold);
                figure(11); plot(corr_line,'b'); hold on;
                for isf = static_frames; plot(isf, corr_line(isf), '*r'); hold on; end; hold off;

                delta_angle   = -0.800;
                static_Nframe = size(static_frames, 2);
                firstframe_flag = 1;

                for frame = 1:static_Nframe
                    tic
                    theta_z = (static_frames(frame)-static_frames(1)) * delta_angle;
                    rotate_x_mat = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
                    rotate_y_mat = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
                    rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0; sind(theta_z) cosd(theta_z) 0 0; 0 0 1 0; 0 0 0 1];
                    trans_mat    = eye(4);
                    afine_mat    = trans_mat * rotate_x_mat * rotate_y_mat * rotate_z_mat;
                    detector_corr = detector_new * afine_mat';

                    pa_data_frame    = gpuArray(single(pa_data(:,:,static_frames(frame))));
                    Points_sensor_all = gpuArray(single(detector_corr(:,1:3)));
                    [pa_img, total_angle_weight] = DualSpeedReconstraction_mex( ...
                        [Ellipse.a,Ellipse.b,Ellipse.c,Ellipse.centerx,Ellipse.centery,Ellipse.centerz], ...
                        Points_sensor_all, Points_img, pa_data_frame, ...
                        single(fs), single(predelay), single(VM_out), single(VM_in), single(R));
                    disp(['xframe: ',num2str(xframe),' yframe: ',num2str(yframe),' frame: ',num2str(frame)]);

                    pa_img1 = gather(pa_img);  total_angle_weight = gather(total_angle_weight);
                    pa_img2 = pa_img1 ./ total_angle_weight;
                    [GaussianMask, ~, ~] = generateGaussianMask({x_range, y_range}, 'Center', [center_x,center_y], 'Sigma', 9);
                    GaussianMask = GaussianMask .* ones(1, 1, size(z_range,2));
                    pa_img3 = pa_img2 .* GaussianMask;
                    if firstframe_flag; pa_ref = pa_img3; end
                    pa_total = pa_total + subplus(pa_img3);
                    firstframe_flag = 0;
                    toc

                    set(h_img9_1,'CData',squeeze(max(pa_img2,[],3)));
                    set(h_img9_2,'CData',squeeze(max(GaussianMask,[],3)));
                    set(h_img9_3,'CData',squeeze(max(pa_total,[],3)));
                    drawnow limitrate;

                    imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],1))), sprintf('zx xframe=%d, yframe=%d, frame=%d.png',xframe,yframe,frame));
                    imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],2))), sprintf('zy xframe=%d, yframe=%d, frame=%d.png',xframe,yframe,frame));
                    imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],3))), sprintf('xy xframe=%d, yframe=%d, frame=%d.png',xframe,yframe,frame));
                end % frame loop
                toc

                SubWinSize_y = imgsize(1)-1;  SubWinSize_x = imgsize(2)-1;
                idx_y = totalsize_y-(yframe-1)*step_length_y*resolution_factor-SubWinSize_y : totalsize_y-(yframe-1)*step_length_y*resolution_factor;
                idx_x = (xframe-1)*step_length_x*resolution_factor+1 : (xframe-1)*step_length_x*resolution_factor+1+SubWinSize_x;
                current_weight = GaussianMask * single(static_Nframe);
                pa_count_total(idx_y,idx_x,:) = pa_count_total(idx_y,idx_x,:) + current_weight;
                pa_img_total(idx_y,idx_x,:)   = pa_img_total(idx_y,idx_x,:)   + pa_total;
                pa_img_total_2 = pa_img_total ./ (pa_count_total + eps);

                imin = min(pa_img_total_2,[],'all');  imax = max(pa_img_total_2,[],'all');
                set(h_img10_1,'CData',squeeze(max(pa_img_total_2,[],1))); set(h_img10_1.Parent,'CLim',[imin,imax]);
                set(h_img10_2,'CData',squeeze(max(pa_img_total_2,[],3))); set(h_img10_2.Parent,'CLim',[imin,imax]);
                set(h_img10_3,'CData',squeeze(max(pa_img_total_2,[],2))); set(h_img10_3.Parent,'CLim',[imin,imax]);
                drawnow limitrate;

                imwrite(mat2gray(squeeze(max(pa_img_total_2(end:-1:1,:,:),[],1))), sprintf('step=%d zx xframe=%d, yframe=%d.png',step_x,xframe,yframe));
                imwrite(mat2gray(squeeze(max(pa_img_total_2(end:-1:1,:,:),[],2))), sprintf('step=%d zy xframe=%d, yframe=%d.png',step_x,xframe,yframe));
                imwrite(mat2gray(squeeze(max(pa_img_total_2(end:-1:1,:,:),[],3))), sprintf('step=%d xy xframe=%d, yframe=%d.png',step_x,xframe,yframe));
            end % yframe loop
        end % xframe loop

        save(['pa_img_fullsize_step', num2str(step_x), '.mat'], 'pa_img_total_2', '-v7.3');

    otherwise
        disp('Error: Undefined reconstruct_mode. Please set reconstruct_mode to 1–13.');
end

%% Final Visualization
figure(3);
subplot(131); imagesc(squeeze(max(pa_img_total_2,[],2))); axis equal tight; colormap gray; colorbar; set(gca,'tickdir','out'); ylabel('X'); xlabel('Z'); title('XZ proj');
subplot(132); imagesc(squeeze(max(pa_img_total_2,[],3))); axis equal tight; colormap gray; colorbar; ylabel('Y'); xlabel('X'); title('XY proj'); set(gca,'tickdir','out');
subplot(133); imagesc(squeeze(max(pa_img_total_2,[],1))); axis equal tight; colormap gray; colorbar; set(gca,'tickdir','out'); ylabel('Y'); xlabel('Z'); title('YZ proj');
