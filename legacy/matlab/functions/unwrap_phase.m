%% This file saves phase of calibration images
% 1. Create Mask By Prewitt
% 2. Solve Phases

function [] = unwrap_phase(targetDir, num_f, num_step,HIGHrLOW,CPSD_thresh, phase_name, r_id)
    %% Load Phase Imgs(Real Dataset:The image is cropped for quicker results)
    roots = {'Calibration','Reconstruction', 'Reconstruction_Polar','Calibration_Polar'};
    phase_root = roots{r_id};
    
    % data_root = ['../',phase_root,'/PSPImg/'];
    data_root = fullfile(targetDir, phase_root, '/PSPImg/');
    disp(data_root)
    % bg_i_root = ['../',phase_root,'/Img/'];
    bg_i_root = fullfile(targetDir, phase_root, '/Img/');
    save_root = fullfile(targetDir, phase_root, '/Phases/');
    % save_root = ['../',phase_root,'/Phases/'];
    if ~exist(save_root,'dir')
        mkdir(save_root)
    end

    use_CPSD = true;
    if CPSD_thresh < 0
        use_CPSD = false;
    end
    
    m=num_f;n=num_step;  %fre+steps 
    
    fileList = dir(fullfile(data_root, '*.bmp')); % 可根据实际情况修改文件扩展名
    fileList = fileList(arrayfun(@(x) ~strcmp(x.name(1),'.'), fileList));

    N=length(fileList)/(m*n);  %文件数
    disp(['共有 ', num2str(N), ' 组数据']);
    
    obj_id = 1:N;
    Img_total=cell(m,n);

    %% Unwrap with PSA & MF-TPU & CPSD(optional)
    for k=obj_id
        for i=1:m
            for j=1:n
                path=[data_root, num2str(k), '_', num2str(i), '_', num2str(j), '.bmp'];
                Img_total{i,j}=imread(path);
                Img_total{i,j}=im2double(Img_total{i,j});
            end
        end

        %% N-PS
        phi = psp_get_phase(Img_total, m, n);   
        phase_unwrapped = psp_unwrap(phi,m,HIGHrLOW);
        if use_CPSD
            mask_path = [bg_i_root, 'mask_',num2str(k),'.bmp'];
            mask = CPSD(mask_path,Img_total, CPSD_thresh);
            if (r_id == 1) || (r_id == 4)
                phase = CPC(phase_unwrapped, mask);
            else
                phase = phase_unwrapped .* mask;
            end
        else
            phase = phase_unwrapped;
        end

        save([save_root,phase_name,'_', num2str(k), '.mat'],'phase');    
        imwrite(mat2gray(phase),[save_root,'i_',phase_name,'_', num2str(k), '.bmp']);    

    end

end



