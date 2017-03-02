% -------------------------------------------------------------------------------------------------------------------------
function [newTargetPosition, bestScale] = tracker_eval(net_x, s_x, scoreId, z_features, x_crops, targetPosition, window, p)
%TRACKER_STEP
%   runs a forward pass of the search-region branch of the pre-trained Fully-Convolutional Siamese,
%   reusing the features of the exemplar z computed at the first frame.
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------
    % forward pass, using the pyramid of scaled crops as a "batch"
    % TODO:eval函数需要仔细查看api
    net_x.eval({p.id_feat_z, z_features, 'instance', x_crops});

    % 返回的response是17 * 17 * 1 * 5的，下面只是将这个“1”去掉
    responseMaps = reshape(net_x.vars(scoreId).value, [p.scoreSize p.scoreSize p.numScale]);

    % 这是resize之后的大小，在论文中提到了，将scoremap 向上resize到一个大小之后会更精确
    responseMapsUP = gpuArray(single(zeros(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp, p.numScale)));
    % Choose the scale whose response map has the highest peak
    if p.numScale>1
        currentScaleID = ceil(p.numScale/2);
        bestScale = currentScaleID;
        bestPeak = -Inf;
        for s=1:p.numScale
            if p.responseUp > 1
                % upsample to improve accuracy
                responseMapsUP(:,:,s) = imresize(responseMaps(:,:,s), p.responseUp, 'bicubic');
            else
                responseMapsUP(:,:,s) = responseMaps(:,:,s);
            end
            thisResponse = responseMapsUP(:,:,s);
            % penalize change of scale
            if s~=currentScaleID, thisResponse = thisResponse * p.scalePenalty; end
            thisPeak = max(thisResponse(:));
            if thisPeak > bestPeak, bestPeak = thisPeak; bestScale = s; end
        end
        responseMap = responseMapsUP(:,:,bestScale);
    else
        responseMap = responseMapsUP;
        bestScale = 1;
    end

    % 得到一个最佳的response map,大小为272 * 272
    % 其实上面的过程是找最佳尺度的过程:-)
    % make the response map sum to 1
    responseMap = responseMap - min(responseMap(:));
    responseMap = responseMap / sum(responseMap(:));

    % 加上cosine window, 与一般的加cosine window的方法不同，这是利用线性插值的方式去做
    % apply windowing
    responseMap = (1-p.wInfluence)*responseMap + p.wInfluence*window;
    [r_max, c_max] = find(responseMap == max(responseMap(:)), 1);
    [r_max, c_max] = avoid_empty_position(r_max, c_max, p);
    % 找到加上cosinewindow之后的最大位置的坐标
    p_corr = [r_max, c_max];

    %TODO: 下面是坐标变换，如何把score里面最大值的坐标变换为原图像中的坐标
    % Convert to crop-relative coordinates to frame coordinates
    % displacement from the center in instance final representation ...
    % 在score层相对于中心的偏移,所以下面要用原来的目标大小加这个东西
    disp_instanceFinal = p_corr - ceil(p.scoreSize*p.responseUp/2);

    % ... in instance input ...
    % 因为在求最终的response的时候，从17upsample到了272，所以要除回去
    % stride为跳的步长, 比如隔几个取一个卷积操作
    disp_instanceInput = disp_instanceFinal * p.totalStride / p.responseUp;
    % ... in instance original crop (in frame coordinates)
    % s_x 是当前的搜索区域的大小，p.instanceSize是网络输入的大小(resize之后的大小)
    disp_instanceFrame = disp_instanceInput * s_x / p.instanceSize;
    % position within frame in frame coordinates
    % 相对于上一帧的目标区域的偏移
    newTargetPosition = targetPosition + disp_instanceFrame;
end

function [r_max, c_max] = avoid_empty_position(r_max, c_max, params)
    if isempty(r_max)
        r_max = ceil(params.scoreSize/2);
    end
    if isempty(c_max)
        c_max = ceil(params.scoreSize/2);
    end
end
