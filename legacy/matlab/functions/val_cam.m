function [] = val_cam(del_idx, n_fre, n_step,calib_param_path)

    %% Check Camera Calibration Results.
    % Section1: Delete extra 3 files and rename the rest
    % Section2: Plot reprojected points to images(Check Img index)
    % Section3. Check the conversion error between reprojected points and camera points conversion

    steps = n_step;fres = n_fre;
    check_root = '../Calibration_Polar/Check/';
    calib_im_path= '../Calibration_Polar/Img/';
    calib_psp_path = '../Calibration_Polar/PSPImg/';

    if ~exist(check_root, 'dir')
        mkdir(check_root);
    end

    fileList = dir(fullfile(calib_psp_path, '*.bmp')); % 可根据实际情况修改文件扩展名
    fileList = fileList(arrayfun(@(x) ~strcmp(x.name(1),'.'), fileList));
    N=length(fileList)/(steps*fres);  %文件数
    rn_flag = (N==23);
    disp(['共有 ', num2str(N), ' 组数据']);
    
    load(calib_param_path);
    % cameraParams = tmp;


    %% rename zone
    if ~isempty(del_idx) % need rename
        % remove unused calibration images
        count = 0;
        for k_id=1:N
            path=[calib_im_path, num2str(k_id), '.bmp'];
            if ismember(k_id,del_idx) % if in del_idx, delete, and record idx
                count = count + 1;
                delete(path);
            else
                new_path = [calib_im_path, num2str(k_id - count), '.bmp'];
                if strcmp(new_path, path)
                    continue
                end
                movefile(path, new_path);
            end
        end

        % remove unused calibration psp images
        count = 0;
        for k_id=1:N
            if ismember(k_id,del_idx) % if in del_idx, delete, and record idx
                for i = 1:steps
                    for j = 1:fres
                        path=[calib_psp_path, num2str(k_id), '_',num2str(j), '_',num2str(i),'.bmp'];
                        delete(path);
                    end
                end
                count = count + 1;
            else
                for i = 1:6
                    for j = 1:4
                        path=[calib_psp_path, num2str(k_id), '_',num2str(j), '_',num2str(i),'.bmp'];
                        new_path=[calib_psp_path, num2str(k_id-count), '_',num2str(j), '_',num2str(i),'.bmp'];
                        if strcmp(new_path, path)
                            continue
                        end
                        movefile(path, new_path)
                    end
                end
            end
        end
    end
    %% Plot reprojected points to images(Check Img index)
    sz = 56; r = sz/8;

    for k_id=1:13
        im_id = k_id;
        path=[calib_im_path, num2str(im_id), '.bmp'];
        img = imread(path);
        p = squeeze(cameraParams.ReprojectedPoints(:,:,k_id));
        imshow(img);hold on;scatter(p(:,1), p(:,2),sz,'filled');
        saveas(gcf, [check_root,num2str(im_id),'.jpg']);
    end

    % save('../Calibration/cameraParams.mat', 'cameraParams');

    %% Relationship between reprojected points and RT (check RT index and Img Idx)

    [M, ~] = size(cameraParams.WorldPoints);
    World_coordinate = [cameraParams.WorldPoints, zeros(M,1)];
    K = cameraParams.IntrinsicMatrix';
    fx = K(1,1);  fy = K(2,2);  u0 = K(1,3);  v0 = K(2,3); s_factor=K(1,2);
    k1=cameraParams.RadialDistortion;	% 相机畸变系数
    k2=cameraParams.TangentialDistortion;
    k=[k1,k2];

    errs = zeros(1,20);
    for k_id = 1:13
        p = squeeze(cameraParams.ReprojectedPoints(:,:,k_id));    
        cam_coord = zeros(3, M);
        for p_id = 1:M
            u = p(p_id, 1); v = p(p_id, 2);
            ydn = (v-v0)/fy;
            xdn = (u-u0-s_factor*ydn)/fx;
            r = sqrt(xdn^2+ydn^2);
            xn = xdn-xdn*(k(1)*r^2+k(2)*r^4);   %注意检查标定工具箱中k的含义
            yn = ydn-ydn*(k(1)*r^2+k(2)*r^4);
            zn = 1;
            cam_coord(:, p_id) = [xn;yn;zn];
        end

        R=cameraParams.RotationMatrices(:,:,k_id);  %注意此项转置
        T=cameraParams.TranslationVectors(k_id,:);  %注意此项转置
        Corner_coordinate = World_coordinate*R+T;  %求取角点在相机坐标系坐标，注意此项转置
        Corner_coordinate = Corner_coordinate ./ Corner_coordinate(:, 3);    

        a = norm(Corner_coordinate - cam_coord');
        errs(1,k_id) = a;

    %     figure;plot(Corner_coordinate(:,1), 'lineWidth',2);hold on;plot(cam_coord(1,:));title('X')
    %     figure;plot(Corner_coordinate(:,2), 'lineWidth',2);hold on;plot(cam_coord(2,:));title('Y')

    end
    figure;plot(Corner_coordinate(:,1), 'lineWidth',2);hold on;plot(cam_coord(1,:));title('X')
    figure;plot(Corner_coordinate(:,2), 'lineWidth',2);hold on;plot(cam_coord(2,:));title('Y')



