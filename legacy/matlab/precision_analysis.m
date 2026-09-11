% cali_main
%% Set parameters
clear; clc; close all;
num_f = 4;
num_step = 6;
HIGHrLOW = [2,4,4];
use_CPSD = true;
% CPSD_thresh = 0.2;
srcDir = '../����/';
dstDir = '../Reconstruction/';

extra = 0;
phase_name = 'PSP';
r_id = 2; % r_id = 2,代表相位路径存在Reconstruction�?

addpath('functions');
%% 1. rename the files
rename(num_f, num_step, srcDir, dstDir, extra);
%% 2. unwrap the phases
CPSD_thresh = 0.4;    % 根据mask结果进行修改，尽量保证mask中只保留球面/平面范围
unwrap_phase(num_f, num_step, HIGHrLOW, CPSD_thresh, phase_name, r_id);

%% 3. phase2pc
%% 3.1 standard ball
% lut_path = '../Calibration/LUTs/LUT_PSP.mat';
lut_path = '../Calibration_Polar/LUTs/LUT_Polar.mat';
pc_save_path = [dstDir, 'pc_data/'];    % 点云数据保存路径
result_save_path = [dstDir, 'PrecisionResults/'];
if ~exist(pc_save_path, 'dir')
    mkdir(pc_save_path);
end
if ~exist(result_save_path, 'dir')
    mkdir(result_save_path);
end

fileList = dir(fullfile([dstDir,'/PSPImg'], '*.bmp'));
N = length(fileList)/(num_f*num_step);  % 文件�?
ball_start_num = 9;    % 球面重建�?始id
ball_end_num = 15;      % 球面重建结束id
zmin = 420;            % z轴（相机光轴方向）重建范围，根据实际情况修改
zmax = 500;

% 球面误差相关变量初始�?
positionNum = ball_end_num-ball_start_num+1;
disp(['共有 ', num2str(positionNum), ' 组球面点云数�?']);
ballPointDisMean = zeros(positionNum, 1);   % positionNum个位置，偏差的均�?
ballPointDisStd = zeros(positionNum, 1);    % positionNum个位置，偏差的std
ballParameters = zeros(positionNum, 4);     % X0,Y0,Z0,R

for k = ball_start_num:ball_end_num
 % for k = 10    
    phase_path = [dstDir,'Phases/',phase_name, '_',num2str(k),  '.mat'];
    mask_path = [dstDir,'Img/', 'mask_',num2str(k),'.bmp'];
    if exist(mask_path, 'file')
        mask = imread(mask_path);
    else
        tmp = load(phase_path);
        phase = tmp.phase';
        mask = ones(size(phase));
    end

    [ptMat, ptCloud] = phase2pc(lut_path, phase_path, mask, zmin, zmax);
    figure;pcshow(ptCloud);title(['Reconstruction result of object ', num2str(k)]);
    pcwrite(ptCloud, [pc_save_path, 'BallPcdata', num2str(k), '.pcd'], 'Encoding', 'ascii'); 
    
    % 球面拟合和去除外�?
    [ballModel, inlierIdx] = pcfitsphere(ptCloud, 1); % RANSAC剔除离群点，1为内点到模型的最大距�?
    ptMat = ptMat(inlierIdx, :);
    % ballModel数据存储
    x = ptMat(:,1);
    y = ptMat(:,2);
    z = ptMat(:,3);
    % 不同位置的点云拟�?
    fun = @(a) (x-a(1)).^2+(y-a(2)).^2+(z-a(3)).^2-a(4)*a(4);    % 拟合参数: (cx,cy,cz) 与半径R 单位mm
    a0 = [1,1,1,1];
    options = optimset('Algorithm','Levenberg-Marquardt','Display','off');
    solution = lsqnonlin(fun,a0,[],[],options);
    if solution(4) < 0
        solution(4) = -solution(4);    % 解算半径为负�?
    end
    idx = k-ball_start_num+1;
    ballParameters(idx, :) = solution;
    % 绘图
%     plot_ellipsoid(ball_parameters,ballData);hold on;
%     figure;scatter3(x,y,z,'.r'); axis equal;
%     figure;pcshow(ballPc);
    % 球度误差
    pointDisList = sqrt((x-solution(1)).^2+...
        (y-solution(2)).^2+(z-solution(3)).^2)-sqrt(solution(4)*solution(4));
    ballPointDisMean(idx) = mean(abs(pointDisList));
    ballPointDisStd(idx) = std(pointDisList);
end
save([result_save_path, 'ballParameters.mat'], 'ballParameters')
save([result_save_path, 'ballPointDisMean.mat'], 'ballPointDisMean');
save([result_save_path, 'ballPointDisStd.mat'], 'ballPointDisStd');
disp(['球面重建误差分析结果已保存至', result_save_path]);

% 与真实�?�比�?
true_R = 20.0185/2;     % 单位:mm
R_error = mean(abs(ballParameters(:, 4) - true_R));
disp(['球面重建半径R平均误差�? ', num2str(R_error)]);
disp(['球面拟合平均误差�? ', num2str(mean(ballPointDisMean))]);
disp(['球面拟合标准偏差误差�? ', num2str(mean(ballPointDisStd))]);

%% 3.2 standard plate
% lut_path = '../Calibration/LUTs/LUT_PSP.mat';
lut_path = '../Calibration_Polar/LUTs/LUT_Polar.mat';
pc_save_path = [dstDir, 'pc_data/'];    % 点云数据保存路径
result_save_path = [dstDir, 'PrecisionResults/'];
if ~exist(pc_save_path, 'dir')
    mkdir(pc_save_path);
end
if ~exist(result_save_path, 'dir')
    mkdir(result_save_path);
end

fileList = dir(fullfile([dstDir,'/PSPImg'], '*.bmp'));
N = length(fileList)/(num_f*num_step);  % 文件�?
plane_start_num = 8;    % 平面重建�?始id
plane_end_num = 15;      % 平面重建结束id
zmin = 420;             % z轴（相机光轴方向）重建范围，根据实际情况修改
zmax = 500;

% 平面误差相关变量初始�?
positionNum = plane_end_num-plane_start_num+1;
disp(['共有 ', num2str(positionNum), ' 组平面点云数�?']);
plane_set_normal = zeros(positionNum, 4);
planePointDisMean = zeros(positionNum, 1); % 偏差的均�?
planePointDisStd = zeros(positionNum, 1);  % 偏差的std

for k = plane_start_num:plane_end_num
% for k = 9    
    phase_path = [dstDir,'Phases/',phase_name, '_',num2str(k),  '.mat'];
    mask_path = [dstDir,'Img/', 'mask_',num2str(k),'.bmp'];
    if exist(mask_path, 'file')
        mask = imread(mask_path);
    else
        tmp = load(phase_path);
        phase = tmp.phase';
        mask = ones(size(phase));
    end

    [ptMat, ptCloud] = phase2pc(lut_path, phase_path, mask, zmin, zmax);
%     [ptMat, ptCloud] = phase2pc_plane(lut_path, phase_path, mask, zmin, zmax);
    figure;pcshow(ptCloud);title(['Reconstruction result of object ', num2str(k)]);
    pcwrite(ptCloud, [pc_save_path, 'PlanePcdata', num2str(k), '.pcd'], 'Encoding', 'ascii'); 

    % 平面拟合
    % Dmat[90000x4 double]，最后一列全�?1，Ax+By+Cz+D=0
    idx = k-plane_start_num+1;
    Dmat = [ ptMat(:,1), ptMat(:,2), ptMat(:,3), ones(length(ptMat(:,1)),1)];
    [U,S,V] = svd(Dmat,0);
    plane_coff = V(:,end);  % smallest singular value A,B,C,D平面参数
    plane_set_normal(idx,:) = plane_coff'/sqrt(sum(plane_coff(1:3).^2));  % 系数归一�?
    if plane_set_normal(idx,1)<0
        plane_set_normal(idx,:) = -plane_set_normal(idx,:);
    end
    % 计算当前位置平面重建点到拟合面的偏差
    % planeData是重建得到的(x,y,z)坐标，如果是标准平面则有Ax0 + By0 + Cz0 + D = 0
    % �?以pointDisList表示当前位置平面重建点到拟合面的偏差
    pointDisList = plane_set_normal(idx,1)*ptMat(:,1)+plane_set_normal(idx,2)*ptMat(:,2)+plane_set_normal(idx,3)*ptMat(:,3)+plane_set_normal(idx,4); %系数已经归一化，分母项为1
    pointDisList(pointDisList>1)=1;     %离群点剔�?
    pointDisList(pointDisList<-1)=-1;   %离群点剔�?
    planePointDisMean(idx) = mean(abs(pointDisList));
    planePointDisStd(idx) = std(pointDisList);
end
save([result_save_path, 'planePointDisMean.mat'], 'planePointDisMean')
save([result_save_path, 'planePointDisStd.mat'], 'planePointDisStd');
disp(['平面重建误差分析结果已保存至', result_save_path]);
disp(['平面拟合平均误差�? ', num2str(mean(planePointDisMean))]);
disp(['平面拟合标准偏差误差�? ', num2str(mean(planePointDisStd))]);


