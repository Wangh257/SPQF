function [planePointDisMean, planePointDisStd] = plane_error(davidData)
    planeNum = 1;
    idx = 1;
    plane_set_normal = zeros(planeNum, 4);
    planePointDisMean = zeros(planeNum, 1); % 5个位置，5个偏差的均值
    planePointDisStd = zeros(planeNum, 1);  % 5个位置，5个偏差的std
    
    Dmat = [ davidData(:,1), davidData(:,2), davidData(:,3),ones(length(davidData(:,1)),1)];    
    [U,S,V] = svd(Dmat,0);
    plane_coff = V(:,end);  % smallest singular value A,B,C,D平面参数
    plane_set_normal(idx,:) = plane_coff'/sqrt(sum(plane_coff(1:3).^2));  % 系数归一化
    if plane_set_normal(idx,1)<0
        plane_set_normal(idx,:) = -plane_set_normal(idx,:);
    end
    % 计算当前位置平面重建点到拟合面的偏差
    % planeData是重建得到的(x,y,z)坐标，如果是标准平面则有Ax0 + By0 + Cz0 + D = 0
    % 所以pointDisList表示当前位置平面重建点到拟合面的偏差
    pointDisList = plane_set_normal(idx,1)*davidData(:,1)+plane_set_normal(idx,2)*davidData(:,2)+plane_set_normal(idx,3)*davidData(:,3)+plane_set_normal(idx,4); %系数已经归一化，分母项为1
%     pointDisList(pointDisList>1)=1;     %离群点剔除
%     pointDisList(pointDisList<-1)=-1;   %离群点剔除
    
%     pointDisList = pointDisList/sqrt(sum(plane_coff(1:3).^2))*plane_coff(3);
    
%     plt_data = davidData;
%     plt_data = plt_data(:,3)-pointDisList;
%     figure;pcshow(pointCloud(plt_data));
    planePointDisMean(idx) = mean(abs(pointDisList));
    planePointDisStd(idx) = std(pointDisList);
    disp('results: pdismean & disstd');
    disp(planePointDisMean);
    disp(planePointDisStd);
end