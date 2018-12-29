% fusion_matrix_multiply.m
%
%	* Created by Vishnu Naresh Boddeti on 5/22/13.
%	* naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
%	* Copyright 2013 Carnegie Mellon University. All rights reserved.

function out = fusion_matrix_multiply(A,B,num_blocks_A,num_blocks_B)

[size_blocks,blocks_B] = size(B);
out = zeros(size_blocks,num_blocks_A(1)*num_blocks_B(2));

index = 1;
for i = 1:num_blocks_A(1)
    ind1 = (i-1)*num_blocks_A(2)+1:i*num_blocks_A(2);
    for j = 1:num_blocks_B(2)
        ind2 = j:num_blocks_B(2):blocks_B;
        out(:,index) = sum(A(:,ind1).*B(:,ind2),2);
        index = index + 1;
    end
end