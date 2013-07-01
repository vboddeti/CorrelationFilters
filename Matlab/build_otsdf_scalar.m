% build_otsdf_scalar.m
% 
%	* Citation: This function calls the correlation filter design proposed in the following paper for scalar features.
%	* Optimal trade-off synthetic discriminant function filters for arbitrary devices, B.V.K.Kumar, D.W.Carlson, A.Mahalanobis - Optics 
%	* Letters, 1994.
%	* Notes: This filter design is good when the dimensionality of the data is greater than the training sample size. Setting alpha to zero results in the famous MACE filter.
%  
%	* Created by Vishnu Naresh Boddeti on 5/22/13.
%	* naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
%	* Copyright 2013 Carnegie Mellon University. All rights reserved.

function otsdf = build_otsdf_scalar(img,args)

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

otsdf.filt_freq = reshape((Y*(K\A))',args.size);
otsdf.filt = ifft_images(otsdf.filt_freq);

for i = 1:size(otsdf.filt,3)
    otsdf.filt(:,:,i) = rot90(otsdf.filt(:,:,i),2);
end

otsdf.shift = 1;