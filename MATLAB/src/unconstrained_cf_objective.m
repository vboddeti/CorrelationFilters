% unconstrained_cf_objective.m
%
% * Created by Vishnu Naresh Boddeti on 9/30/14.
% * naresh@cmu.edu (http://vishnu.boddeti.net)
% * Copyright 2014 Carnegie Mellon University. All rights reserved.

function val = unconstrained_cf_objective(H,T,p)

dim = sqrt(size(T,2));
temp = fusion_matrix_multiply(T,H,[dim,dim],[dim,1]);
val = sum(fusion_matrix_multiply(conj(H),temp,[1,dim],[dim,1]));
val = val - 2*sum(fusion_matrix_multiply(conj(H),p,[1,dim],[dim,1]));
val = real(val);