% pre_whiten_data_fusion.m
%
%   * Created by Vishnu Naresh Boddeti on 5/22/13.
%   * naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
%   * Copyright 2013 Carnegie Mellon University. All rights reserved.

function [X,Y,S,Sinv] = pre_whiten_data_fusion(img,args)

alpha = args.alpha;
beta = args.beta;
siz = args.size;

num_img = length(img);
[~,~,dim] = size(img(1).im);
d = siz(1)*siz(2);

S = zeros(d,dim*dim);
X = fft_images_fusion(img, siz);

ind = [];

for i = 1:num_img
    index = 1;
    for p = 1:dim
        for q = 1:dim
            tmp = conj(X(:,:,p,i)).*X(:,:,q,i);
            S(:,index) = S(:,index) + tmp(:);
            if p == q
                ind = [ind,index];
            end
            index = index + 1;
        end
    end
end

ind = unique(ind);
S = S/max(S(:));
S = beta*S;
S(:,ind) = alpha*ones(siz(1)*siz(2),dim)+S(:,ind);

Sinv = fusion_matrix_inverse(S,[dim,dim]);
Y = zeros(size(X));

for i = 1:num_img
    tmp = reshape(X(:,:,:,i),[siz(1)*siz(2),dim]);
    tmp = fusion_matrix_multiply(Sinv,tmp,[dim,dim],[dim,1]);
    Y(:,:,:,i) = reshape(tmp,[siz(1),siz(2),dim]);
end