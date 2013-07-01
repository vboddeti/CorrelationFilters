% build_uotsdf_scalar.m
%
%	* Citation: This function calls the correlation filter design proposed in the following paper for scalar features.
%	* Optimal trade-off synthetic discriminant function filters for arbitrary devices, B.V.K.Kumar, D.W.Carlson, A.Mahalanobis - Optics 
%	* Letters, 1994.
%	* Notes: This is currently the fastest Correlation Filter design to train, and is highly amenable for real-time online learning or 
%	* for object tracking. This is very similar to the MOSSE filter,
%	* D. S. Bolme, J. R. Beveridge, B. A. Draper, and Y. M. Lui. Visual Object Tracking using Adaptive Correlation Filters, CVPR. June 2010
%
%	* Created by Vishnu Naresh Boddeti on 5/22/13.
%	* naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
%	* Copyright 2013 Carnegie Mellon University. All rights reserved.

function uotsdf = build_uotsdf_scalar(img,args)

num_img = length(img);
[~,ind] = sort([img.label],'descend');
img = img(ind);
A = -ones(num_img,1);
A([img.label]==1) = 1;

[~,Y] = pre_whiten_data(img,args);
clear img;

num_pos_img = sum(A==1);
num_neg_img = sum(A==-1);

uotsdf.filt_freq = mean(Y(:,:,A==1),3)/num_pos_img - mean(Y(:,:,A==-1),3)/num_neg_img;
uotsdf.filt = ifft_images(uotsdf.filt_freq);

uotsdf.shift = 1;