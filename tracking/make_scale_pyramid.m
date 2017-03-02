% -----------------------------------------------------------------------------------------------------
function pyramid = make_scale_pyramid(im, targetPosition, in_side_scaled, out_side, avgChans, stats, p)
%MAKE_SCALE_PYRAMID
%   computes a pyramid of re-scaled copies of the target (centered on TARGETPOSITION)
%   and resizes them to OUT_SIDE. If crops exceed image boundaries they are padded with AVGCHANS.
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -----------------------------------------------------------------------------------------------------
    in_side_scaled = round(in_side_scaled);
    pyramid = gpuArray(zeros(out_side, out_side, 3, p.numScale, 'single'));
    max_target_side = in_side_scaled(end);
    min_target_side = in_side_scaled(1);
    % 这个不是太懂
    % TODO：为什么下面的beta乘以最大的搜索区域就能得到搜索区域。。？
    % 这是为了保证，当将搜索区域resize的时候，都是从大到小resize，而不是从小到大resize
    % 这样图片能够保留更多的信息，更准确一些；但是也可能导致搜索区域过大，包含了很多的不必要的背景信息
    beta = out_side / min_target_side;
    % size_in_search_area = beta * size_in_image
    % e.g. out_side = beta * min_target_side
    search_side = round(beta * max_target_side);
    % 首先将最大的区域裁减下来，然后再把小的裁出来
    [search_region, ~] = get_subwindow_tracking(im, targetPosition, [search_side search_side], [max_target_side max_target_side], avgChans);
    if p.subMean
        search_region = bsxfun(@minus, search_region, reshape(stats.x.rgbMean, [1 1 3]));
    end
    assert(round(beta * min_target_side)==out_side);

    for s = 1:p.numScale
        target_side = round(beta * in_side_scaled(s));
        pyramid(:,:,:,s) = get_subwindow_tracking(search_region, (1+search_side*[1 1])/2, [out_side out_side], target_side*[1 1], avgChans);
    end
end
