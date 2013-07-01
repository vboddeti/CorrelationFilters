% fft_images.m
%
%	* Created by Vishnu Naresh Boddeti on 5/22/13.
%	* naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
%	* Copyright 2013 Carnegie Mellon University. All rights reserved.

function X = fft_images(img,siz)

num_img = length(img);
X = zeros(siz(1),siz(2),num_img);

for i = 1:num_img
    X(:,:,i) = fft2(img(i).im,siz(1),siz(2))/sqrt(siz(1)*siz(2));
end