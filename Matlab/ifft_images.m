% ifft_images.m
%
%	* Created by Vishnu Naresh Boddeti on 5/22/13.
%	* naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
%	* Copyright 2013 Carnegie Mellon University. All rights reserved.

function X = ifft_images(img)

[m,n,num_img] = size(img);
X = zeros(m,n,num_img);

for i = 1:num_img
    X(:,:,i) = ifft2(img(:,:,i),'symmetric')/sqrt(m*n);
end