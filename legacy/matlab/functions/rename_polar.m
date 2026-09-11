%% This file is used to rename the calibration images
function [] = rename_polar(num_f, num_step, srcDir, dstDir, extra)
   channel = 4;
    %% create dirs
    im_dir = [dstDir, '/Img/'];
    pim_dir = [dstDir, '/PolarPSPImg/'];
    if ~exist(dstDir , 'dir')
        mkdir(dstDir);
    end
    if ~exist(im_dir, 'dir')
        mkdir(im_dir);
    end
    if ~exist(pim_dir, 'dir')
        mkdir(pim_dir);
    end
    for c = 1:channel
        cim_dir = [pim_dir,'Channel',num2str(c)];
        if ~exist(cim_dir , 'dir')
            mkdir(cim_dir);
        end
    end
    
    %% extract indentifier
    p_num = num_f*num_step + 1;
    if extra == 0
        p_num = num_f*num_step;
    end
    fileList = dir(fullfile(srcDir, '*.bmp')); % 修改为图片格式的扩展名
    fileList = fileList(arrayfun(@(x) ~strcmp(x.name(1),'.'), fileList));
    % 从fileList结构中提取名称
    names = {fileList.name};
    [~, filenames] = cellfun(@(x) fileparts(x), names, 'UniformOutput', false);
    % 使用正则表达式提取名称的数字部分
    reg_format = '\d{9}_\d{4}';
    numericValues = cellfun(@(x) regexp(x,reg_format,'match'), filenames);

    % 按照数字部分的升序对文件名进行排序
    [~, sortedIndices] = sort(numericValues);

    num=length(fileList)/p_num;
    disp(['共有 ', num2str(num/channel), ' 组数据']);

    % move to dirs
    for img_idx=1:length(fileList) 
        srcImN = fullfile(srcDir, fileList(img_idx).name);
        % disp(srcImN)
        srcIm = imread(srcImN);

        n = ceil(img_idx/p_num);
        subgroup = mod(n,channel);
        if subgroup == 0
            subgroup = 4;
        end
        
        %% no blank img
        if extra == 0
            tmp = img_idx - (n-1)*p_num;
            i = ceil(tmp/num_step);
            j = tmp - (i-1)*num_step;
            obj_idx = ceil(n/channel);
            dstImN = sprintf('%d_%d_%d.bmp', obj_idx, i, j);
            cim_dir = [pim_dir,'Channel',num2str(subgroup),'/'];
            dstImP = [cim_dir,dstImN];
            imwrite(srcIm, dstImP);
            continue;
        end
        %% have blank img
        if mod(img_idx, p_num) == 0
            dstImN = sprintf('%d.bmp', n);
            dstImP = [im_dir,dstImN];
            if subgroup == 1
                imwrite(srcIm, dstImP);
            end
        else
            tmp = img_idx - (n-1)*p_num;
            i = ceil(tmp/num_step);
            j = tmp - (i-1)*num_step;
            
            obj_idx = ceil(n/channel);
            dstImN = sprintf('%d_%d_%d.bmp', obj_idx, i, j);
            cim_dir = [pim_dir,'Channel',num2str(subgroup),'/'];
            dstImP = [cim_dir,dstImN];
            imwrite(srcIm, dstImP);
        end
    end
end
