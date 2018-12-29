% build_uotsdf.m
%
% * Citation: This function calls the correlation filter design proposed in the following paper for vector features.
% * Vishnu Naresh Boddeti, Takeo Kanade and B. V. K. Vijaya Kumar, "Correlation Filters for Object Alignment," CVPR 2013
% * Notes: This is currently the fastest Correlation Filter design to train, and is highly amenable for real-time online learning or
% * for object tracking.
%
% * Created by Vishnu Naresh Boddeti on 5/22/13.
% * naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
% * Copyright 2013 Carnegie Mellon University. All rights reserved.

function out = build_uotsdf

global D;
global args;

if ~args.psd_flag
    compute_psd;
    args.psd_flag = 1;
end

S = compute_inverse_psd;
mean_filt = compute_mean;
filt_freq = pre_whiten_data(S,mean_filt);

out.b = 0;
out.filt_freq = h_prox(filt_freq,args);
tmp = ifft2(out.filt_freq,'symmetric')*prod(args.fft_size);
out.filt = tmp(1:args.img_size(1),1:args.img_size(2),:);
out.args = args;