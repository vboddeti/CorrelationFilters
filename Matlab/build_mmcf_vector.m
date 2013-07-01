% build_mmcf_vector.m
%  
%	* Citation: This function calls the correlation filter design proposed in the following PhD thesis.
%	* Vishnu Naresh Boddeti, "Advances in Correlation Filters: Vector Features, Structured Prediction and Shape Alignment" PhD thesis, 
%	* Carnegie Mellon University, Pittsburgh, PA, USA, 2012.
%	* Notes: This currently the best performing Correlation Filter design for vector features, especially when the training sample size is 
%	* larger than the dimensionality of the data.
%  
%	* Created by Vishnu Naresh Boddeti on 5/22/13.
%	* naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
%	* Copyright 2013 Carnegie Mellon University. All rights reserved.

function mmvcf = build_mmcf_vector(img,args)

num_img = length(img);
[~,ind] = sort([img.label],'descend');
img = img(ind);
A = -ones(num_img,1);
A([img.label]==1) = 1;

[X,Y] = pre_whiten_data_fusion(img,args);
clear img;

X = reshape(X,[prod(args.size),num_img]);
Y = reshape(Y,[prod(args.size),num_img]);

X1 = (real(X));
X2 = (imag(X));
clear X;

Y1 = (real(Y));
Y2 = (imag(Y));

K = X1'*Y1;
clear X1;
clear Y1;
K = K + X2'*Y2;
clear X2;
clear Y2;

K = (K'+K)/2;
K = real(K);

K = [(1:num_img)' K];

param = ['-s 0 -t 4 -c ' num2str(args.C) ' -w1 ' num2str(args.wpos)];
mmvcf_model = svmtrain(A,K,param);

Y = Y(:,mmvcf_model.SVs);
mmvcf_weights = Y*mmvcf_model.sv_coef;

mmvcf.filt_freq = reshape(mmvcf_weights',args.size);
mmvcf.filt = ifft_images(mmvcf.filt_freq);

for i = 1:size(mmvcf.filt,3)
    mmvcf.filt(:,:,i) = rot90(mmvcf.filt(:,:,i),2);
end

mmvcf.b = mmvcf_model.rho;
mmvcf.shift = 1;