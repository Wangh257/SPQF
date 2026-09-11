function save_cloud_path = WH_recc(sourceDir, targetDir, calDir)
    % cali_main
    %% Set parameters
    num_f = 4;
    num_step = 12;
    HIGHrLOW = [2,4,4];
    use_CPSD = true;
    CPSD_thresh = 0.20;
    [currentDir,~,~] = fileparts(mfilename('fullpath'));
    % srcDir = '../物体偏振/';
    srcDir = fullfile(sourceDir);
    dstDir = fullfile(targetDir, 'Reconstruction_Polar/');
    % dstDir = '../Reconstruction_Polar/';
    save_cloud_dir = [dstDir, 'cloud_result/']
    if ~exist(save_cloud_dir, 'dir')
            mkdir(save_cloud_dir);
    end
    disp('this hanshu')
    phase_name = 'Polar_PSP'; % 相位的名称，如Polar_PSP_1.mat
    extra = 0;
    r_id = 3; % 3代表相位路径存储在Reconstruction_Polar下
    % addpath('functions');
    [currentDir,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(currentDir, 'functions'));
    %% 1. rename the files
    rename_polar(num_f, num_step, srcDir, dstDir, extra);
    
    %% 2. fuse imgs
    fileList = dir(fullfile([dstDir,'PolarPSPImg/Channel1'], '*.bmp')); % 可根据实际情况修改文件扩展名
    fileList = fileList(arrayfun(@(x) ~strcmp(x.name(1),'.'), fileList));
    N=length(fileList)/(num_f*num_step);  %文件数
    for obj_id = 1:N
        fuse_channel(dstDir, obj_id, num_f, num_step);
    end
    %% 3. unwrap the phases
    unwrap_phase(targetDir, num_f, num_step,HIGHrLOW, CPSD_thresh, phase_name, r_id);
    
    %% 4. phase2pc
    % lut_path = '../Calibration_Polar/LUTs/LUT_Polar.mat';
    lut_path = fullfile(calDir, 'Calibration_Polar/LUTs/LUT_Polar.mat')
    zmin = 500;
    zmax = 700;
    for k = 1:N
        phase_path = [dstDir,'Phases/',phase_name, '_',num2str(k),  '.mat'];
        mask_path = [dstDir,'Img/', 'mask_',num2str(k),'.bmp'];
        save_cloud_path = [save_cloud_dir, '_', num2str(k), '.ply'];
    
        if exist(mask_path, 'file')
            mask = imread(mask_path);
        else
            tmp = load(phase_path);
            phase = tmp.phase';
            mask = ones(size(phase));
        end
    
        [~, ptCloud] = phase2pc(lut_path, phase_path,mask,zmin,zmax);
        pcwrite(ptCloud, save_cloud_path);
        
    end
    % figure;pcshow(ptCloud);title(['Reconstruction result of object ', num2str(k)]);
end
