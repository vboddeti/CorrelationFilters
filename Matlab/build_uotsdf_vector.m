% build_uotsdf_vector.m
%  
%	* Citation: This function calls the correlation filter design proposed in the following paper for vector features.
%	* Vishnu Naresh Boddeti, Takeo Kanade and B. V. K. Vijaya Kumar, "Correlation Filters for Object Alignment," CVPR 2013
%	* Notes: This is currently the fastest Correlation Filter design to train, and is highly amenable for real-time online learning or 
%	* for object tracking.
%  
%	* Created by Vishnu Naresh Boddeti on 5/22/13.
%	* naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
%	* Copyright 2013 Carnegie Mellon University. All rights reserved.

function uotsdf = build_uotsdf_vector(img,args)

num_img = length(img);
[~,ind] = sort([img.label],'descend');
img = img(ind);
A = -ones(num_img,1);
A([img.label]==1) = 1;

[~,Y] = pre_whiten_data_fusion(img,args);
clear img;

num_pos_img = sum(A==1);
num_neg_img = sum(A==-1);

uotsdf.filt_freq = mean(Y(:,:,:,A==1),4)/num_pos_img - mean(Y(:,:,:,A==-1),4)/num_neg_img;
uotsdf.filt = ifft_images_fusion(uotsdf.filt_freq);

uotsdf.shift = 1;