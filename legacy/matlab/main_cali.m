% cali_main
%% Set parameters
num_f = 4;
num_step = 6;
HIGHrLOW = [4,4,4];
CPSD_thresh = 0.2; % if thresh <0, do not use CPSD
srcDir = '../标定';
dstDir = '../Calibration';

phase_name = 'PSPImg';% 相位的名称为PSP_1.mat ... etc.
r_id = 1; % r_id = 1,代表相位路径存在Calibration下

addpath('functions');
%% 1. rename the files
rename(num_f, num_step, srcDir, dstDir);
%% 2. fetch calibration parameters;check parameters and rename the files
cameraCalibrator
%% save params
camera_param_path = '../Calibration/cameraParams.mat';
save(camera_param_path, 'cameraParams');
% if your calibration imgs is less than 23, do not use below codes
del_idx = [11,12,21];
val_cam(del_idx, num_f, num_step,camera_param_path);

%% 3. unwrap the phases
unwrap_phase(num_f, num_step,HIGHrLOW,CPSD_thresh,phase_name,r_id);
%% 4. get LUT
LUT_name = 'LUT_PSP';
calib_dir = 'Calibration';
get_lut_cali(calib_dir, camera_param_path, LUT_name,phase_name)