% build_mmcf_dual.m
%
% * Citation: This function implements the correlation filter design proposed in the publications.
% * A. Rodriguez, Vishnu Naresh Boddeti, B.V.K. Vijaya Kumar and A. Mahalanobis, "Maximum Margin Correlation Filter: A New Approach for
% * Localization and Classification", IEEE Transactions on Image Processing, 2012.
% * Vishnu Naresh Boddeti and B.V.K. Vijaya Kumar, "Maximum Margin Vector Correlation Filter" Arxiv 1404.6031 (April 2014)
% * Vishnu Naresh Boddeti, "Advances in Correlation Filters: Vector Features, Structured Prediction and Shape Alignment" PhD thesis,
% * Carnegie Mellon University, Pittsburgh, PA, USA, 2012.
% * Notes: This currently the best performing Correlation Filter design, especially when the training sample size is
% * larger than the dimensionality of the data.
%
% * Created by Vishnu Naresh Boddeti on 5/22/13.
% * naresh@cmu.edu (http://vishnu.boddeti.net)
% * Copyright 2013 Carnegie Mellon University. All rights reserved.

function out = build_mmcf_dual

global args;
global labels;
global data_freq;

if ~args.psd_flag
    compute_psd;
    args.psd_flag = 1;
end

S = compute_inverse_psd;
[m,n,dim,num] = size(data_freq);

Y = pre_whiten_data(S,data_freq);
data_freq = reshape(data_freq,[m*n*dim,num]);
Y = reshape(Y,[m*n*dim,num]);
K = data_freq'*Y;
K = (K+K')/2;
K = double(real(K));
% K = K/max(max(abs(K)));
K = [(1:num)' K];
options = ['-s 0 -t 4 -c ' num2str(args.C)];
mmcf = svmtrain(labels,K,options);
b = -mmcf.rho;
w = Y(:,mmcf.sv_indices)*mmcf.sv_coef;
w = reshape(w,[m,n,dim]);
data_freq = reshape(data_freq,[m,n,dim,num]);

out.b = b;
out.filt_freq = h_prox(w,args);
tmp = ifft2(out.filt_freq,'symmetric')*prod(args.fft_size);
out.filt = tmp(1:args.img_size(1),1:args.img_size(2),:);
out.args = args;
