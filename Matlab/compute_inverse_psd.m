% compute_inverse_psd.m
%
% * Created by Vishnu Naresh Boddeti on 9/30/14.
% * naresh@cmu.edu (http://vishnu.boddeti.net)
% * Copyright 2014 Carnegie Mellon University. All rights reserved.

function S = compute_inverse_psd

global D;
global args;

alpha = args.alpha;
beta = args.beta;
[d,dim] = size(D);
dim = sqrt(dim);
ind = 1:(dim+1):dim^2;
S = beta*D;
S(:,ind) = alpha*ones(d,dim)+S(:,ind);
ptr = matWrapper(size(S));
ptr.Data = S;
inds = (1:dim^2)';
inds = reshape(inds, [dim dim])';
fusion_matrix_inverse(ptr,inds);
S = ptr.Data;