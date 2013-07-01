% fusion_matrix_inverse.m
%
%   * Created by Vishnu Naresh Boddeti on 5/22/13.
%   * naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
%   * Copyright 2013 Carnegie Mellon University. All rights reserved.

function X = fusion_matrix_inverse(X,num_blocks)

if (num_blocks(1) == 2) && (num_blocks(2) == 2)
    DC = X(:,3)./X(:,4);
    BD = X(:,2)./X(:,4);
    BDC = X(:,2).*DC;
    ABDC = 1./(X(:,1)-BDC);
    X(:,1) = ABDC;
    X(:,2) = -ABDC.*BD;
    X(:,3) = -DC.*ABDC;
    X(:,4) = 1./X(:,4) + DC.*ABDC.*BD;
else
    [~,total_blocks] = size(X);
    ind_D = total_blocks(end);
    ind_B = (num_blocks(2)):num_blocks(2):(total_blocks-num_blocks(2));
    start = (num_blocks(1)-1)*num_blocks(2)+1;
    ind_C = start:(start+num_blocks(2)-1)-1;
    tmp = union(ind_B,ind_D);
    tmp = union(tmp,ind_C);
    tmp = union(tmp,total_blocks-1);
    ind_A = setdiff(1:total_blocks,tmp);
    D = 1./X(:,end);
    val = sqrt(length(ind_A));
    DC = fusion_matrix_multiply(D,X(:,ind_C),[1,1],[1,val]);
    BD = fusion_matrix_multiply(X(:,ind_B),D,[val,1],[1,1]);
    BDC = fusion_matrix_multiply(X(:,ind_B),DC,[val,1],[1,val]);
    tmp = X(:,ind_A)-BDC;
    clear BDC;
    ABDC = fusion_matrix_inverse(tmp,[val,val]);
    
    X(:,ind_A) = ABDC;
    X(:,ind_B) = -fusion_matrix_multiply(ABDC,BD,[val,val],[val,1]);
    X(:,ind_C) = -fusion_matrix_multiply(DC,ABDC,[1,val],[val,val]);
    tmp = D + fusion_matrix_multiply(fusion_matrix_multiply(DC,ABDC,...
        [1,val],[val,val]),BD,[1,val],[val,1]);
    X(:,end) = tmp(:,1);
end

clear D;
clear DC;
clear BD;
clear BDC;
clear ABCD;