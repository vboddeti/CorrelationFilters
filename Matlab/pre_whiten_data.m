% pre_whiten_data.m
%
%	* Created by Vishnu Naresh Boddeti on 5/22/13.
%	* naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
%	* Copyright 2013 Carnegie Mellon University. All rights reserved.

function [X,Y,S] = pre_whiten_data(X,args)

alpha = args.alpha;
beta = args.beta;
siz = args.size;

num_img = length(X);
d = siz(1)*siz(2);
X = fft_images(X, siz);

D = mean(abs(X).^2,3)*d;
D = D/max(D(:));
S = alpha+beta*D;
S = S/max(S(:));
Y = X./repmat(S,[1,1,num_img]);
X(1,1,:) = 0.0;
Y(1,1,:) = 0.0;