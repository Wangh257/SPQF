%% define path
function [] = get_lut_cali(calib_dir, calib_param_path, LUT_name, phase_name)
    phase_path = ['../',calib_dir,'/Phases/'];
    save_root = ['../',calib_dir,'/LUTs/'];
    save_path = [save_root, LUT_name, '.mat'];

    S = load(calib_param_path);
    K = S.cameraParams.IntrinsicMatrix;K=K';  %读取相机内参，注意检查内参矩阵中的偏斜系数s=0？
    [M, tmp] = size(S.cameraParams.WorldPoints);
    

    fx = K(1,1);  fy = K(2,2);  u0 = K(1,3);  v0 = K(2,3); s_factor=K(1,2);
    k1 = S.cameraParams.RadialDistortion;	% 相机畸变系数
    k2 = S.cameraParams.TangentialDistortion;
    k = [k1,k2];
    
    %%
    N = size(S.cameraParams.RotationMatrices, 3);
    R=zeros(3,3,N);T=zeros(1,3,N);
    for i=1:N
        R(:,:,i)=S.cameraParams.RotationMatrices(:,:,i);  %注意此项转置
        T(:,:,i)=S.cameraParams.TranslationVectors(i,:);  %注意此项转置
    end
    %2)加载角点世界坐标
    World_coordinate = [S.cameraParams.WorldPoints, zeros(M,1)];
    %3)加载相位图
    for i = 1:N
        path = [phase_path, phase_name, '_',num2str(i), '.mat'];
        tmp = load(path);
        img_phases(i, : ,:) = tmp.phase';
    end
    [~, img_width, img_height] = size(img_phases); %注意左侧索引顺序为(u,v),此处有转置

    %%%%%%%%%%%%%%%%%%%%用角点求解N个平面的方程
    Corner_coordinate=zeros(M,3,N);	% 角点坐标，M为每张标定图角点个数49，N为图像个数
    calibraPlane=zeros(N,4);		% 20个平面方程：Ax+By+Cz+D=0
    for i=1:N
        for j=1:M
            Corner_coordinate(:,:,i)=World_coordinate*R(:,:,i)+T(:,:,i);  %求取角点在相机坐标系坐标，注意此项转置
        end
        % D是任一张平面上M个角点的坐标（X_c, Y_c, Z_c）
        D = [ Corner_coordinate(:,1,i), Corner_coordinate(:,2,i), Corner_coordinate(:,3,i),ones(size(Corner_coordinate(:,1,i)))];	% 补1成齐次矩阵
        [U,~,V] = svd(D,0);	% svd法解平面方程（相机坐标系下）
        calibraPlane(i,:) = V(:,end)';  % 平面系数 [A B C D],注意归一化
    end

    %%%%%%%%%%%%%%%%%%%%求解直线方程
    line = zeros(2,img_width,img_height);
    for v=1:img_height
        for u=1:img_width
            ydn = (v-v0)/fy;
            xdn = (u-u0-s_factor*ydn)/fx;
            r = sqrt(xdn^2+ydn^2);
            xn = xdn-xdn*(k(1)*r^2+k(2)*r^4);   %注意检查标定工具箱中k的含义
            yn = ydn-ydn*(k(1)*r^2+k(2)*r^4);
            zn = 1;
            % 在距离光心距离为1的平面
            line(:,u,v)=[xn/zn;yn/zn];   %注意普通矩阵比CELL运算快	
        end
    end

    % 直线与平面的交点，共有N*img_width*img_height个交点(Xc,Yc,Zc)
    % 相机坐标系（Oc Xc Yc）外部一点与Oc连接，交归一化平面（即图像坐标系-到光心Ox距离为f）
    % 一点的坐标总共20*H*W个，再四频六步相移法求出每个点的绝对相位，作标定即可
    Camera_coordinate=zeros(N,3,img_width,img_height);
    for v=1:img_height
        for u=1:img_width
            Camera_coordinate(:,3,u,v)=-calibraPlane(:,4)./(calibraPlane(:,1)*line(1,u,v)+calibraPlane(:,2)*line(2,u,v)+calibraPlane(:,3));
            Camera_coordinate(:,1,u,v)=line(1,u,v)*Camera_coordinate(:,3,u,v);
            Camera_coordinate(:,2,u,v)=line(2,u,v)*Camera_coordinate(:,3,u,v); 
        end
    end

    % 多项式共有3×4个标定参数
    LUT_Poly = zeros(3,4,img_width,img_height);
    % 使用最小二乘法
    for v=1:img_height
        for u=1:img_width
            phases_uv=img_phases(:,u,v);    % uv对应的N个相位
            coor_uv=Camera_coordinate(:,:,u,v);    % Camera_coordinate(N,3,img_width,img_height)表示射线与平面的交点坐标
%             % 标定Xc中的参数a1,a2,a3,a4
%             A = [ones(size(phases_uv)), phases_uv, phases_uv.^2, phases_uv.^3];
%             b = [coor_uv(:,1)];
%             LUT_Poly(1,:,u,v) = (A'*A)/(A'*b);
%             % 标定Yc中的参数b1,b2,b3,b4
%             b = [coor_uv(:,2)];
%             LUT_Poly(2,:,u,v) = (A'*A)/(A'*b);
%             % 标定Zc中的参数c1,c2,c3,c4
%             b = [coor_uv(:,3)];
%             LUT_Poly(3,:,u,v) = (A'*A)/(A'*b);

            % modify:上述可合并为下式矩阵求解，标定Xc中的参数a1,a2,a3,a4
            A = [ ones(size(phases_uv)), phases_uv, phases_uv.^2, phases_uv.^3];
            LUT_Poly(:,:,u,v) = coor_uv'/A';
        end
    end
    if ~exist(save_root,'dir')
        mkdir(save_root);
    end
    save(save_path, 'LUT_Poly');
    disp(['LUT poly has been saved to ', save_path])
    %% LUT pixel fitting
    load(save_path)
    % point_u = 220;point_v=202;
    point_u = 500;point_v=300;
    lp = min(img_phases(:,point_u,point_v));
    up = max(img_phases(:,point_u,point_v));
    phiList = lp-5:up+5;
    % 标定Zc中的参数c1,c2,c3,c4，从phi映射到轴
    ZcList = LUT_Poly(3,1,point_u,point_v)+LUT_Poly(3,2,point_u,point_v)*phiList+LUT_Poly(3,3,point_u,point_v)*phiList.^2+LUT_Poly(3,4,point_u,point_v)*phiList.^3;
    plot(phiList, ZcList,'b-','LineWidth',2);
    hold on;scatter(img_phases(:,point_u,point_v),Camera_coordinate(:,3,point_u,point_v),'r','filled');hold off;
    label = legend('Ploy fitting curve','real data');label.set('FontSize',12)
    xlabel('phase/rad','FontSize',14);
    ylabel('Z/mm','FontSize',14);
    %set(gca,'XLim',[0 120]);
    zmin=560;
    zmax=700;
    %% Reconstruction Test
    for k = 1
        phase_path = ['../',calib_dir,'/Phases/',phase_name, '_',num2str(k),  '.mat'];
        mask_path = ['../',calib_dir,'/Img/', 'mask_',num2str(k),'.bmp'];

        
        if exist(mask_path, 'file')
            mask = imread(mask_path);
        else
            tmp = load(phase_path);
            phase = tmp.phase';
            mask = ones(size(phase));
        end
        
        [~, ptCloud] = phase2pc(save_path, phase_path, mask,zmin,zmax);
    end
    figure;pcshow(ptCloud);title(['Reconstruction result of ',num2str(k), ' plane']);
    