% cali_main
%% Set parameters
num_f = 4;
num_step = 6;
HIGHrLOW = [4,4,4];
use_CPSD = true;
CPSD_thresh = 0.2;
srcDir = '../物体';
dstDir = '../Reconstruction';
phase_name = 'PSP';
r_id = 2; % r_id = 2,代表相位路径存在Reconstruction下

[currentDir,~,~] = fileparts(mfilename('fullpath'));
addpath(fullfile(currentDir, 'functions'));
%% 1. rename the files
rename(num_f, num_step, srcDir, dstDir);

%% 2. unwrap the phases
unwrap_phase(num_f, num_step,HIGHrLOW, CPSD_thresh, phase_name, r_id);

%% 3. phase2pc
phase_root = 'Reconstruction';
lut_path = '../Calibration/LUTs/LUT_PSP.mat';
for k = 2
    phase_path = ['../',phase_root,'/Phases/',phase_name, '_',num2str(k),  '.mat'];
    mask_path = ['../',phase_root,'/Img/', 'mask_',num2str(k),'.bmp'];

    if exist(mask_path, 'file')
        mask = imread(mask_path);
    else
        tmp = load(phase_path);
        phase = tmp.phase';
        mask = ones(size(phase));
    end

    [~, ptCloud] = phase2pc(lut_path, phase_path,mask);
end
figure;pcshow(ptCloud);title(['Reconstruction result of object ', num2str(k)]);