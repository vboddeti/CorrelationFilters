% compute_mean.m
%
% * Created by Vishnu Naresh Boddeti on 9/30/14.
% * naresh@cmu.edu (http://vishnu.boddeti.net)
% * Copyright 2014 Carnegie Mellon University. All rights reserved.

function mean_filt = compute_mean

global args;
global data;
global labels;
global data_freq;

fft_scale_factor = sqrt(prod(args.fft_size));

num = max(size(data,4),size(data_freq,4));
pos_labels = gaussian_shaped_labels(args.target_magnitude, args.target_sigma, args.img_size(1:2));
neg_labels = -args.target_magnitude * ones(args.img_size(1:2));

% pos_labels = pos_labels - mean2(pos_labels);
neg_labels = neg_labels - mean2(neg_labels);

pos_labels = fft2(pos_labels, args.fft_size(1), args.fft_size(2))/fft_scale_factor;
neg_labels = fft2(neg_labels, args.fft_size(1), args.fft_size(2))/fft_scale_factor;

pos_labels(1,1) = 0;

if isempty(data_freq)
    mean_pos = 0;
    mean_neg = 0;
    for i = 1:num
        if labels(i) == 1
            mean_pos = mean_pos + fft2(data(:,:,:,i),args.fft_size(1),args.fft_size(2));
        end
        if labels(i) == -1
            mean_neg = mean_neg + fft2(data(:,:,:,i),args.fft_size(1),args.fft_size(2));
        end
    end
    mean_pos = mean_pos.*repmat(pos_labels,[1,1,args.dim]);
    mean_neg = mean_neg.*repmat(neg_labels,[1,1,args.dim]);
    mean_filt = mean_pos + mean_neg;
    mean_filt = mean_filt/fft_scale_factor;
else
    mean_pos = sum((data_freq(:,:,:,labels==1)),4).*repmat(pos_labels,[1,1,args.dim]);
    if sum(labels==-1) > 0
        mean_neg = sum((data_freq(:,:,:,labels==-1)),4).*repmat(neg_labels,[1,1,args.dim]);
    else
        mean_neg = 0;
    end
    mean_filt = mean_pos + mean_neg;
end
mean_filt = mean_filt/sum(labels==1);
mean_filt = mean_filt/numel(pos_labels);