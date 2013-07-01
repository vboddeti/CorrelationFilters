% fft_images_fusion.m
%
%	* Created by Vishnu Naresh Boddeti on 5/22/13.
%	* naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
%	* Copyright 2013 Carnegie Mellon University. All rights reserved.

function X = fft_images_fusion(img,siz)

num_img = length(img);
[~,~,dim] = size(img(1).im);
X = zeros(siz(1),siz(2),dim,num_img);

for i = 1:num_img
    for j = 1:dim
        tmp = img(i).im(:,:,j);
        X(:,:,j,i) = fft2(tmp,siz(1),siz(2))/sqrt(siz(1)*siz(2));
    end
end