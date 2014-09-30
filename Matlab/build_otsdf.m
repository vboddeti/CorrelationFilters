% build_otsdf.m
%  
%	* Citation: This function calls the correlation filter design proposed in the following PhD thesis and paper.
%	* Jason Thornton, "Matching deformed and occluded iris patterns: a probabilistic model based on discriminative cues." PhD thesis, 
%	* Carnegie Mellon University, Pittsburgh, PA, USA, 2007.
%	* Vishnu Naresh Boddeti, Jonathon M Smereka, and B. V. K. Vijaya Kumar, "A comparative evaluation of iris and ocular recognition methods
%	* on challenging ocular images." IJCB, 2011
%	* Notes: This filter design is good when the dimensionality of the data is greater than the training sample size.
%  
%	* Created by Vishnu Naresh Boddeti on 5/22/13.
%	* naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
%	* Copyright 2013 Carnegie Mellon University. All rights reserved.

function out = build_otsdf

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

b = 0;
w = (Y*(K\labels));
w = reshape(w,[m,n,dim]);
data_freq = reshape(data_freq,[m,n,dim,num]);

out.b = b;
out.filt_freq = h_prox(w,args);
tmp = ifft2(out.filt_freq,'symmetric')*prod(args.fft_size);
out.filt = tmp(1:args.img_size(1),1:args.img_size(2),:);
out.args = args;
