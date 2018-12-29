% pre_whiten_data.m
%
% * Created by Vishnu Naresh Boddeti on 9/30/14.
% * naresh@cmu.edu (http://vishnu.boddeti.net)
% * Copyright 2014 Carnegie Mellon University. All rights reserved.

function img = pre_whiten_data(S,img)

[m,n,dim,num] = size(img);

for i = 1:num
    tmp = reshape(img(:,:,:,i),[m*n,dim]);
    tmp = fusion_matrix_multiply(S,tmp,[dim,dim],[dim,1]);
    img(:,:,:,i) = reshape(tmp,[m,n,dim]);
end