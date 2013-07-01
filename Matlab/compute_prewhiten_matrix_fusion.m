% compute_prewhiten_matrix_fusion.m
%
%	* Created by Vishnu Naresh Boddeti on 5/22/13.
%	* naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
%	* Copyright 2013 Carnegie Mellon University. All rights reserved.

function [S,ind] = compute_prewhiten_matrix_fusion(X)

[x,y,dim,num_img] = size(X);
d = x*y;

S = zeros(d,dim*dim);
ind = [];

index = 1;
for p = 1:dim
    for q = 1:dim
        tmp = mean(conj(X(:,:,p,:)).*X(:,:,q,:),4);
        S(:,index) = tmp(:);
        if p == q
            ind = [ind,index];
        end
        index = index + 1;
    end
end

ind = unique(ind);
S = S/num_img;