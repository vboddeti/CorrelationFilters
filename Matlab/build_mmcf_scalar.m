% build_mmcf_scalar.m
% 
%	* Citation: This function implements the correlation filter design proposed in the following paper.
%	* A. Rodriguez, Vishnu Naresh Boddeti, B.V.K. Vijaya Kumar and A. Mahalanobis, "Maximum Margin Correlation Filter: A New Approach for
% 	* Localization and Classification", IEEE Transactions on Image Processing, 2012.
% 	* Notes: This currently the best performing Correlation Filter design for scalar features, especially when the training sample size is 
%	* larger than the dimensionality of the data.
% 
%	* Created by Vishnu Naresh Boddeti on 5/22/13.
%	* naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
%	* Copyright 2013 Carnegie Mellon University. All rights reserved.

function mmscf = build_mmcf_scalar(img,args)

num_img = length(img);
[~,ind] = sort([img.label],'descend');
img = img(ind);
A = -ones(num_img,1);
A([img.label]==1) = 1;

[X,Y] = pre_whiten_data(img,args);
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
mmscf_model = svmtrain(A,K,param);

Y = Y(:,mmscf_model.SVs);
mmscf_weights = Y*mmscf_model.sv_coef;
mmscf.filt_freq = reshape(mmscf_weights,args.size);
mmscf.filt = ifft_images(mmscf.filt_freq);

mmscf.b = mmscf_model.rho;
mmscf.shift = 1;