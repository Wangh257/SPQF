function [g_pmax_k, p_max] = max_not_saturate(imgs_seq, ratio)
%UNTITLED 此处显示有关此函数的摘要
% 此处显示详细说明
% 输入: 
%   imgs_seq已按平均灰度大小升序排列
% 输出:
%   g_pmax_k: 对第k张条纹图像拍到的M张偏振图进行merge，每个像素取最大不饱和通道形成的图像
%   p_max:    论文中的p_max = argmax{g_p_k}

height = size(imgs_seq, 1);
width = size(imgs_seq, 2);
g_pmax_k = zeros(height, width);
p_max = zeros(height, width);
M = size(imgs_seq, 3);  % 偏振通道数
% 取最大不饱和通道，对M张偏振图进行融合
for i = 1:height
    for j = 1:width
        cur_p_channel = M;  % 当前偏振通道
        while imgs_seq(i,j,cur_p_channel) >= ratio && cur_p_channel > 1
            cur_p_channel = cur_p_channel - 1;
        end 
        p_max(i, j) = cur_p_channel;    % 默认选亮度最大的偏振通道
        g_pmax_k(i, j) = imgs_seq(i, j, cur_p_channel);
    end
end



