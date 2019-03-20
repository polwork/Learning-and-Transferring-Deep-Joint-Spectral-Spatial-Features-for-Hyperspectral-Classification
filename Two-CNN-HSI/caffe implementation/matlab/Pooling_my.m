function [ pooledFeature ] = Pooling_my( poolDim_h, poolDim_w, stride, convolvedFeature )
% Pools the given convolved features
%
% Parameters:
%  poolDim_h - height of pooling region
%  poolDim_w - width of pooling region
%  stride    - stride of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     
% Author : Jingxiang Yang
% Date : May, 2016

[convolvedFeature_h, convolvedFeature_w] = size(convolvedFeature);

pooled_h = ceil((convolvedFeature_h - poolDim_h) / stride) + 1;
pooled_w = ceil((convolvedFeature_w - poolDim_w) / stride) + 1;
pooledFeature = single(zeros(pooled_h, pooled_w));

for row = 1 : pooled_h
    for col = 1 : pooled_w
        hstart = (row - 1) * stride + 1;
        wstart = (col - 1) * stride + 1;
        hend = min ( hstart + poolDim_h - 1, convolvedFeature_h );
        wend = min ( wstart + poolDim_w - 1, convolvedFeature_w );
        pooledFeature ( row, col ) = max( max ( convolvedFeature ( hstart : hend, wstart : wend ) ) );
    end
end

