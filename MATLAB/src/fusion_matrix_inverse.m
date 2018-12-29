% fusion_matrix_inverse.m
%
% Memory efficient inverse of the fusion matrix
% required for designing vector correlation filters.
%
% ptr is a handle object class with a matrix in the field named Data.
% It is used as a workaround to ensure a pass-by-ref to the recursive
% calls. You can use the matWrapper class to create ptr.
%
% * Created by Vishnu Naresh Boddeti on 9/30/14.
% * naresh@cmu.edu (http://vishnu.boddeti.net)
% * Copyright 2014 Carnegie Mellon University. All rights reserved.

function [] = fusion_matrix_inverse(ptr, indices)

num_blocks = size(indices);
if num_blocks(1) ~= num_blocks(2)
    disp('Something Wrong');
    return;
else
    num_blocks = num_blocks(1);
end
if num_blocks == 1
    ptr.Data = 1./ptr.Data;
    return;
end

if (num_blocks == 2)
    DC = ptr.Data(:,indices(2,1))./ptr.Data(:,indices(2,2));
    BD = ptr.Data(:,indices(1,2))./ptr.Data(:,indices(2,2));
    BDC = ptr.Data(:,indices(1,2)).*DC;
    ABDC = 1./(ptr.Data(:,indices(1,1))-BDC);
    ptr.Data(:,indices(1,1)) = ABDC;
    ptr.Data(:,indices(1,2)) = -ABDC.*BD;
    ptr.Data(:,indices(2,1)) = -DC.*ABDC;
    ptr.Data(:,indices(2,2)) = 1./ptr.Data(:,indices(2,2)) + DC.*ABDC.*BD;
else
    ind_D = indices(end,end);
    ind_B = indices(1:end-1,end);
    ind_C = indices(end,1:end-1);
    ind_A = indices(1:end-1,1:end-1)';
    
    D = 1./ptr.Data(:,ind_D);
    val = length(ind_A);
    DC = fusion_matrix_multiply(D,ptr.Data(:,ind_C),[1,1],[1,val]);
    BD = fusion_matrix_multiply(ptr.Data(:,ind_B),D,[val,1],[1,1]);
    BDC = fusion_matrix_multiply(ptr.Data(:,ind_B),DC,[val,1],[1,val]);
    ptr.Data(:,ind_A) = ptr.Data(:,ind_A)-BDC;
    clear BDC;
    fusion_matrix_inverse(ptr, ind_A);
    ptr.Data(:,ind_B) = -fusion_matrix_multiply(ptr.Data(:,ind_A),BD,[val,val],[val,1]);
    ptr.Data(:,ind_C) = -fusion_matrix_multiply(DC,ptr.Data(:,ind_A),[1,val],[val,val]);
    tmp = D + fusion_matrix_multiply(fusion_matrix_multiply(DC,ptr.Data(:,ind_A),...
        [1,val],[val,val]),BD,[1,val],[val,1]);
    ptr.Data(:,ind_D) = tmp(:,1);
end