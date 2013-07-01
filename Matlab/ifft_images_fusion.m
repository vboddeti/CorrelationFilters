% ifft_images_fusion.m
%
%	* Created by Vishnu Naresh Boddeti on 5/22/13.
%	* naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
%	* Copyright 2013 Carnegie Mellon University. All rights reserved.

function X = ifft_images_fusion(img)

[m,n,dim,num_img] = size(img);
X = zeros(m,n,dim,num_img);

for i = 1:num_img
    for j = 1:dim
        X(:,:,j,i) = ifft2(img(:,:,j,i),'symmetric')/sqrt(m*n);
    end
end