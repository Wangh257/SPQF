function [raw_data, ptCloud] = phase2pc(LUT_path, phase_path, mask_path,zmin,zmax)

    LUT_Poly_Struct = load(LUT_path);
    LUT_Poly = LUT_Poly_Struct.LUT_Poly;
    [~, ~, im_w, im_h] = size(LUT_Poly);
    
    if size(phase_path, 1) == im_h
        phase = phase_path';
    elseif size(phase_path, 1) == im_w
        phase = phase_path;
    else
        tmp = load(phase_path);
        phase = tmp.phase';
    end
    
    
    if numel(mask_path) == 0
        mask = ones(im_w, im_h);
    elseif size(mask_path, 1) == im_h
        mask = mask_path';
    elseif size(mask_path, 1) == im_w
        mask = mask_path;
    else
        mask = logical(imread(mask_path));
        mask = mask';
    end



    data=zeros(im_h*im_w,3);
    for v_idx=1:im_h
        for u_idx=1:im_w
            tempIdx = im_w*(v_idx-1)+u_idx;
            if mask(u_idx, v_idx) == 1
                p1 = phase(u_idx,v_idx);
                p2 = p1*p1;
                p3 = p1*p2;
                ps = [1; p1;p2;p3];
                
                for k = 1:3
                    data(tempIdx,k) = LUT_Poly(k,:,u_idx,v_idx)*ps;
                end
            else
                data(tempIdx, :) = nan;
            end
        end
    end
    raw_data = data;
    data(all(data==0,2),:) = []; % 去除所有为0的行
    data(data(:, 3) > zmax | data(:, 3) < zmin, :) = [];

    data = data(all(~isnan(data), 2), :);    % 去掉为nan的数据
    ptCloud = pointCloud(data);

end