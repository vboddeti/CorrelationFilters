% fusion_matrix_inverse_inplace.m
% 
% Memory efficient inverse of the fusion matrix 
% required for designing vector correlation filters.
% We use the schur complement to perform the matrix inverse.
% 

function [] = fusion_matrix_inverse_inplace(indices)

global X;
num_blocks = size(indices);

if (num_blocks(1) == 2) && (num_blocks(2) == 2)
    DC = X(:,indices(2,1))./X(:,indices(2,2));
    BD = X(:,indices(1,2))./X(:,indices(2,2));
    BDC = X(:,indices(1,2)).*DC;
    ABDC = 1./(X(:,indices(1,1))-BDC);
    X(:,indices(1,1)) = ABDC;
    X(:,indices(1,2)) = -ABDC.*BD;
    X(:,indices(2,1)) = -DC.*ABDC;
    X(:,indices(2,2)) = 1./X(:,indices(2,2)) + DC.*ABDC.*BD;
else
    ind_D = indices(end,end);
    ind_B = indices(1:end-1,end);
    ind_C = indices(end,1:end-1);
    ind_A = indices(1:end-1,1:end-1)';
    
    D = 1./X(:,ind_D);
    val = length(ind_A);
    DC = fusion_matrix_multiply(D,X(:,ind_C),[1,1],[1,val]);
    BD = fusion_matrix_multiply(X(:,ind_B),D,[val,1],[1,1]);
    BDC = fusion_matrix_multiply(X(:,ind_B),DC,[val,1],[1,val]);
    X(:,ind_A) = X(:,ind_A)-BDC;
    clear BDC;
    fusion_matrix_inverse_inplace(ind_A);
    X(:,ind_B) = -fusion_matrix_multiply(X(:,ind_A),BD,[val,val],[val,1]);
    X(:,ind_C) = -fusion_matrix_multiply(DC,X(:,ind_A),[1,val],[val,val]);
    tmp = D + fusion_matrix_multiply(fusion_matrix_multiply(DC,X(:,ind_A),...
        [1,val],[val,val]),BD,[1,val],[val,1]);
    X(:,ind_D) = tmp(:,1);
end

clear D;
clear DC;
clear BD;
clear BDC;
clear ABCD;