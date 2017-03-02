% -------------------------------------------------------------------------------------------------
function bboxes = tracker(varargin)
%TRACKER
%   is the main function that performs the tracking loop
%   Default parameters are overwritten by VARARGIN
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------
    % These are the default hyper-params for SiamFC-3S
    % The ones for SiamFC (5 scales) are in params-5s.txt
    p.numScale = 3;
    p.scaleStep = 1.0375;
    p.scalePenalty = 0.9745;
    p.scaleLR = 0.59; % damping factor for scale update
    p.responseUp = 16; % upsampling the small 17x17 response helps with the accuracy
    p.windowing = 'cosine'; % to penalize large displacements
    p.wInfluence = 0.176; % windowing influence (in convex sum)
    p.net = '2016-08-17.net.mat';
    %% execution, visualization, benchmark
    p.video = 'vot15_bag';
    p.visualization = false;
    p.gpus = 1;
    p.bbox_output = false;
    p.fout = -1;
    %% Params from the network architecture, have to be consistent with the training
    p.exemplarSize = 127;  % input z size
    p.instanceSize = 255;  % input x size (search region)
    p.scoreSize = 17;
    p.totalStride = 8;
    p.contextAmount = 0.5; % context amount for the exemplar
    p.subMean = false;
    %% SiamFC prefix and ids
    p.prefix_z = 'a_'; % used to identify the layers of the exemplar
    p.prefix_x = 'b_'; % used to identify the layers of the instance
    p.prefix_join = 'xcorr';
    p.prefix_adj = 'adjust';
    p.id_feat_z = 'a_feat';
    p.id_score = 'score';
    % Overwrite default parameters with varargin
    p = vl_argparse(p, varargin);
% -------------------------------------------------------------------------------------------------

    % Get environment-specific default paths.
    p = env_paths_tracking(p);
    % Load ImageNet Video statistics
    if exist(p.stats_path,'file')
        stats = load(p.stats_path);
    else
        warning('No stats found at %s', p.stats_path);
        stats = [];
    end
    % Load two copies of the pre-trained network
    % 加载同样的网络到net_z和net_x里面，这个网络包含了siamese的全部部分，之后再在各自的网络里把多余的删掉
    net_z = load_pretrained([p.net_base_path p.net], p.gpus);
    net_x = load_pretrained([p.net_base_path p.net], []);
    [imgFiles, targetPosition, targetSize] = load_video_info(p.seq_base_path, p.video);
    nImgs = numel(imgFiles);
    startFrame = 1;
    % Divide the net in 2
    % exemplar branch (used only once per video) computes features for the target
    % 将多余的网络删掉
    remove_layers_from_prefix(net_z, p.prefix_x);
    remove_layers_from_prefix(net_z, p.prefix_join);
    remove_layers_from_prefix(net_z, p.prefix_adj);
    % instance branch computes features for search region x and cross-correlates with z features
    remove_layers_from_prefix(net_x, p.prefix_z);
    % net_z是计算第一帧的特征的，它对应的输出的feature的id是zFeatId
    zFeatId = net_z.getVarIndex(p.id_feat_z);
    % net_x的输入不仅用x，还有z的feature，直接输出相似性分数
    scoreId = net_x.getVarIndex(p.id_score);

    % get the first frame of the video
    im = gpuArray(single(imgFiles{startFrame}));
    % if grayscale repeat one channel to match filters size
	if(size(im, 3)==1)
        im = repmat(im, [1 1 3]);
    end
    % Init visualization
    videoPlayer = [];
    if p.visualization && isToolboxAvailable('Computer Vision System Toolbox')
        videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im,2), size(im,1)]+30]);
    end
    % get avg for padding
    avgChans = gather([mean(mean(im(:,:,1))) mean(mean(im(:,:,2))) mean(mean(im(:,:,3)))]);

    % wc_z和hc_z是代表原来的目标区域，加上适当的背景区域之后的长宽
    wc_z = targetSize(2) + p.contextAmount*sum(targetSize);
    hc_z = targetSize(1) + p.contextAmount*sum(targetSize);
    s_z = sqrt(wc_z*hc_z);
    % examplarSize 是论文中提到的127，而s_z是padd区域之后的长乘以宽开根号
    % 然后就得到了从目标大小到网络输入大小的scale变换
    scale_z = p.exemplarSize / s_z;

    % initialize the exemplar
    % 得到第一帧的crop区域,这个crop区域已经是examplar大小了
    [z_crop, ~] = get_subwindow_tracking(im, targetPosition, [p.exemplarSize p.exemplarSize], [round(s_z) round(s_z)], avgChans);
    if p.subMean
        z_crop = bsxfun(@minus, z_crop, reshape(stats.z.rgbMean, [1 1 3]));
    end
    % TODO: d_search代表的是？
    d_search = (p.instanceSize - p.exemplarSize)/2;
    pad = d_search/scale_z;
    s_x = s_z + 2*pad;
    % arbitrary scale saturation
    min_s_x = 0.2*s_x;
    max_s_x = 5*s_x;

    switch p.windowing
        case 'cosine'
            window = single(hann(p.scoreSize*p.responseUp) * hann(p.scoreSize*p.responseUp)');
        case 'uniform'
            window = single(ones(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp));
    end
    % make the window sum 1
    window = window / sum(window(:));
    scales = (p.scaleStep .^ ((ceil(p.numScale/2)-p.numScale) : floor(p.numScale/2)));

    % evaluate the offline-trained network for exemplar z features
    % 将第一帧的图像输入网络，得到第一帧的特征
    net_z.eval({'exemplar', z_crop});
    z_features = net_z.vars(zFeatId).value;
    z_features = repmat(z_features, [1 1 1 p.numScale]);

    bboxes = zeros(nImgs, 4);
    % start tracking
    tic;
    for i = startFrame:nImgs
        if i>startFrame
            % load new frame on GPU
            im = gpuArray(single(imgFiles{i}));
   			% if grayscale repeat one channel to match filters size
    		if(size(im, 3)==1)
        		im = repmat(im, [1 1 3]);
    		end
            % scaledInstance 在当前帧搜索区域经过scale变换之后的大小
            scaledInstance = s_x .* scales;

            % scaledTarget 在当前帧经过scale变换之后的目标大小
            scaledTarget = [targetSize(1) .* scales; targetSize(2) .* scales];

            % 裁减位于targetPosition(上一帧的目标中心的位置)的几个patch，具体做法见该文件注释
            % extract scaled crops for search region x at previous target position
            x_crops = make_scale_pyramid(im, targetPosition, scaledInstance, p.instanceSize, avgChans, stats, p);

            % 将上面裁减出来的x_crops与第一帧的特征一起送入网络，返回响应值最大的目标中心以及对应的尺度变换index
            % evaluate the offline-trained network for exemplar x features
            [newTargetPosition, newScale] = tracker_eval(net_x, round(s_x), scoreId, z_features, x_crops, targetPosition, window, p);
            targetPosition = gather(newTargetPosition);

            % scale damping and saturation
            % 更新scale之后的尺度以及目标的大小，
            % 跟一般的不太一样，都是采用了线性差值的方式
            s_x = max(min_s_x, min(max_s_x, (1-p.scaleLR)*s_x + p.scaleLR*scaledInstance(newScale)));
            targetSize = (1-p.scaleLR)*targetSize + p.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
        else
            % at the first frame output position and size passed as input (ground truth)
        end

        rectPosition = [targetPosition([2,1]) - targetSize([2,1])/2, targetSize([2,1])];
        % output bbox in the original frame coordinates
        oTargetPosition = targetPosition; % .* frameSize ./ newFrameSize;
        oTargetSize = targetSize; % .* frameSize ./ newFrameSize;
        bboxes(i, :) = [oTargetPosition([2,1]) - oTargetSize([2,1])/2, oTargetSize([2,1])];

        if p.visualization
            if isempty(videoPlayer)
                figure(1), imshow(im/255);
                figure(1), rectangle('Position', rectPosition, 'LineWidth', 4, 'EdgeColor', 'y');
                drawnow
                fprintf('Frame %d\n', startFrame+i);
            else
                im = gather(im)/255;
                im = insertShape(im, 'Rectangle', rectPosition, 'LineWidth', 4, 'Color', 'yellow');
                % Display the annotated video frame using the video player object.
                step(videoPlayer, im);
            end
        end

        if p.bbox_output
            fprintf(p.fout,'%.2f,%.2f,%.2f,%.2f\n', bboxes(i, :));
        end

    end

    bboxes = bboxes(startFrame : i, :);

end
