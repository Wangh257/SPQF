% cali_main
%% Set parameters
num_f = 4;
num_step = 12;
HIGHrLOW = [4,4,4];
CPSD_thresh = 0.25; % if thresh <0, do not use CPSD
srcDir = '../cal';
dstDir = '../Calibration_Polar';
currentDir = pwd;
% 获取当前的父目录
[parentDir,~,~] = fileparts(currentDir);

phase_name = 'PSP';
r_id = 4; % r_id = 1,代表相位路径存在Calibration下

addpath('functions');
%% 1. rename the files
rename(num_f, num_step, srcDir, dstDir);

%% 2. fetch calibration parameters;check parameters and rename the files
% if you already have cameraParams, just set the path
% else, use
cameraCalibrator
%% save
camera_param_path = '../Calibration_Polar/cameraParams.mat';
save(camera_param_path, 'cameraParams');
% del_idx = [11];
% val_cam(del_idx, num_f, num_step,camera_param_path);
%% 3. unwrap the phases
unwrap_phase(parentDir, num_f, num_step,HIGHrLOW,CPSD_thresh,phase_name,r_id);
%% 4. get LUT
LUT_name = 'LUT_Polar';
calib_dir = 'Calibration_Polar';
get_lut_cali(calib_dir, camera_param_path, LUT_name,phase_name)