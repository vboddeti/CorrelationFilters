% unconstrained_cf_gradient.m
%
% * Created by Vishnu Naresh Boddeti on 9/30/14.
% * naresh@cmu.edu (http://vishnu.boddeti.net)
% * Copyright 2014 Carnegie Mellon University. All rights reserved.

function grad = unconstrained_cf_gradient(H,T,p)
dim = sqrt(size(T,2));
grad = 2*fusion_matrix_multiply(T,H,[dim,dim],[dim,1]) - 2*p;