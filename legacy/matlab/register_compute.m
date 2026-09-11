function [registered_pcd,rmse,min_dis,max_dis,mid_dis] = register_compute(sourse_pcd_path,target_pcd_path, save_pcd_path)

% 参数说明
% input
% sourse_pcd:源点云
% target_pcd:目标点云
% output
% registered_pcd:配准后点云
% rmse:配准点云与目标点云的均方根误差RMSE
% min_dis:配准点云与目标点云的最小误差
% max_dis:配准点云与目标点云的最大误差
% mid_dis:配准点云与目标点云的误差中值

    %降采样
    gridSize = 0.5; % 设置降采样网格大小'
    sourse_pcd = pcread(sourse_pcd_path);
    target_pcd = pcread(target_pcd_path);
    sourse_ds = pcdownsample(sourse_pcd, 'gridAverage', gridSize);
    target_ds = pcdownsample(target_pcd, 'gridAverage', gridSize);
    
    sourse_ds = init_pc(sourse_ds);
    target_ds = init_pc(target_ds);

    [rmse,mid_dis,dis] = compute(sourse_ds, target_ds);
    min_dis = min(dis);
    max_dis = max(dis);

    % 使用ICP算法进行配准
    icp_N = 2;  % ICP迭代次数
    for i = 1:icp_N
        [~, registered_pcd] = pcregistericp(sourse_ds, target_ds);
        sourse_ds = registered_pcd;

        [rmse,mid_dis,dis] = compute(sourse_ds, target_ds);
        min_dis = min(dis);
        max_dis = max(dis);

        if(min_dis < 0.1)
            break;
        end

    end
    pcwrite(registered_pcd, save_pcd_path);
    % 可视化配准后的点云
    % figure;
    % pcshowpair(target_ds, registered_pcd);
    % title('Registered Point Clouds');

end


% 计算均方根误差（RMSE）函数
function [rmse,mid,dis] = compute(ptCloud1, ptCloud2)
    % 确保两个点云具有相同数量的点
    numPts = min(ptCloud1.Count, ptCloud2.Count);
    
    % 计算每个点的距离平方
    distancesSquared = sum((ptCloud1.Location(1:numPts,:) - ptCloud2.Location(1:numPts,:)).^2, 2);
    dis = sqrt(distancesSquared);

    % 计算均方根误差
    rmse = sqrt(mean(distancesSquared));
    mid = median(dis);
    
end


% 调整点云初始位置并滤除突变值
function ptCloud  = init_pc(pc)
%     ptCloud = [];
    % 调整点云数据的坐标
    ptCloud = pc.Location;
    ptCloud(:,1) = ptCloud(:,1)-min(ptCloud(:,1));
    ptCloud(:,2) = ptCloud(:,2)-min(ptCloud(:,2));
    ptCloud(:,3) = ptCloud(:,3)-min(ptCloud(:,3));

%     % 使用中位值滤波去除噪声和突变值
%     window_size = 50; % 滤波窗口大小
%     ptCloud(:,3) = medfilt1(ptCloud(:,3), window_size);

    ptCloud = pointCloud(ptCloud);

end
