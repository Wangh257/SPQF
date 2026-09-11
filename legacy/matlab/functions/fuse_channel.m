%% img_fusion.m 
%{ 
function: 对相机拍摄存储的图像进行重新命名并保存在指定路径
适用于Balser相机保存图像的命名格式，不适用大恒图像默认命名格式
date:   23/11/05
input:  待修改图像的文件夹路径
        修改后存储rename图像的文件夹路径
output: 在指定文件夹保存重命名后图像，格式为"num_fre_step.bmp"
%}
% -----------------------------------------------------------%

function [] = fuse_channel(rec_dir, obj_id, m, n)
    % addpath('functions'); % 添加包含子函数的文件夹路径
    [currentDir,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(currentDir, 'functions'));
    % srcDir = '../Reconstruction_Polar/PolarPSPImg/';
    srcDir = fullfile(rec_dir, 'PolarPSPImg/')
    % tmpDir = '../Reconstruction_Polar/Fused/';
    tmpDir = fullfile(rec_dir, 'Fused/')
    % dstDir = '../Reconstruction_Polar/PSPImg/';
    dstDir = fullfile(rec_dir, 'PSPImg/')
  
    src_img = sprintf('%sChannel%d/%d_%d_%d.bmp', srcDir, 1, obj_id, 1, 1);
    [im_h, im_w] = size(imread(src_img));
    channels = 4;
    saturate_ratio = 0.95;  % 超过最大光强(此处默认为255)*saturate_ratio视为饱和
    pmax_seq = zeros(im_h, im_w, m*n);   % 每一张条纹图都有个pmax
    
    %% 1.首先计算出同一条纹图不同偏振通道的融合图像
    
    if ~exist(tmpDir, 'dir')
        mkdir(tmpDir);
    end
    
    imgs_seq = zeros(im_h, im_w, channels);
    imgs_sort = zeros(im_h, im_w, channels);
    for i = 1:m
        for j = 1:n
            dst_img = sprintf('%s%d_%d_%d.bmp',tmpDir, obj_id, i, j);
            for c = 1:channels
               src_img = sprintf('%sChannel%d/%d_%d_%d.bmp', srcDir, c, obj_id, i, j);
               imgs_seq(:,:,c) = im2double(imread(src_img));
            end
            %% begin fuse
            means = zeros(channels, 1);
            for t = 1:channels
                means(t) = mean(mean(imgs_seq(:, :, t)));    % mean(A)多维数组，沿大小>1的第一个数组维度计算，将其视为向量
            end
            [~, idx] = sort(means);     % 默认升序
            for t = 1:channels
                imgs_sort(:, :, t) = imgs_seq(:, :, idx(t));
            end
            [imgs_polarmax_seq, pmax] = max_not_saturate(imgs_sort, saturate_ratio);   % 取最大不饱和的通道
            pmax_seq(:,:,n*(i-1)+j) = pmax;
            
            want_to_save_result = 1;
            if (want_to_save_result)
                imwrite(imgs_polarmax_seq, dst_img);
            end
            
        end
    end
    
    %% 2. 计算决策图
    decision_map = zeros(im_h, im_w, m);     % 每一频的N步都有1个decision_map
    for i = 1:m
        temp_map = zeros(im_h, im_w, n);
        for j = 1:n
            temp_map(:,:,j) = pmax_seq(:,:,n*(i-1)+j);
        end
        [cur_map, ~] = min(temp_map, [], 3);
        decision_map(:,:,i) = cur_map;

        % image(cur_map,'CDataMapping','scaled'); colorbar;
        % saveas(gcf,[tmpDir, 'dmap_',num2str(obj_id),'_', num2str(i), '.png']);        
    end
    
    %% 3. 根据decision_map进行merge融合图像，gk = g_p_k
    if ~exist(dstDir, 'dir')
        mkdir(dstDir);
    end

    for i = 1:m
        cur_map = decision_map(:,:,i);  % 读取第m频的N步相移对应采用的偏振通道decision_map
        for j = 1:n
            % 指定输入输出文件位置
            merged_img = zeros(im_h, im_w);
            dst_img = sprintf('%s%d_%d_%d.bmp',dstDir, obj_id, i, j);
            for c = 1:channels
               src_img = sprintf('%sChannel%d/%d_%d_%d.bmp',srcDir, c, obj_id, i, j);
               imgs_seq(:,:,c) = im2double(imread(src_img));
            end

            means = zeros(channels, 1);
            for t = 1:channels
                means(t) = mean(mean(imgs_seq(:, :, t)));    % mean(A)多维数组，沿大小>1的第一个数组维度计算，将其视为向量
            end
            [~, idx] = sort(means);     % 默认升序
            for t = 1:channels
                imgs_sort(:, :, t) = imgs_seq(:, :, idx(t));
            end

            for ii = 1:im_h
                for jj = 1:im_w
                    merged_img(ii,jj) = imgs_sort(ii,jj,cur_map(ii,jj)); % 融合图像
                end
            end
            % 保存最后合成图像
            imwrite(merged_img, dst_img);   
        end
    end
end







