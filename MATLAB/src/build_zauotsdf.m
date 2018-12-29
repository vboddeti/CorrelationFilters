% build_zauotsdf.m;
%
% * Citation: This function implements the zero-aliasing version of UOTSDF in the following publication.
% * Joseph Fernandez, Vishnu Naresh Boddeti, Andres Rodriguez and B.V.K. Vijaya Kumar, "Zero-Aliasing Correlation Filters for Object Recognition", 
% * IEEE Transactions on Pattern Analysis and Machine Intelligence (Under Review).
% * Notes: This currently the best performing Correlation Filter design, especially when the training sample size is
% * larger than the dimensionality of the data.
%
% * Created by Vishnu Naresh Boddeti on 9/30/14.
% * naresh@cmu.edu (http://vishnu.boddeti.net)
% * Copyright 2014 Carnegie Mellon University. All rights reserved.

function out = build_zauotsdf

global D;
global args;

if ~args.psd_flag
    compute_psd;
    args.psd_flag = 1;
end

D = D/max(abs(D(:)));
alpha = args.alpha;
beta = args.beta;
[d,dim] = size(D);
dim = sqrt(dim);
ind = 1:(dim+1):dim^2;
T = beta*D;
T(:,ind) = alpha*ones(d,dim)+T(:,ind);
fft_scale_factor = prod(args.fft_size);

p = compute_mean;
[m,n,dim] = size(p);
fft_size = [m,n,dim];

if ~isfield(args,'h_base')
    S = compute_inverse_psd;
    H_base = pre_whiten_data(S,p);
else
    H_base = fft2(args.h_base,args.fft_size(1),args.fft_size(2))/fft_scale_factor;
end
[~,H,hspatial_base] = h_prox(H_base,args);
p = reshape(p,[m*n,dim]);
H = reshape(H,[m*n,dim]);

%% ZACF Proximal Gradient Descent Iterations

stop = 0;
rel_err_vect = [];
tlist = [];
H_old = H;

obj_val_list = real(unconstrained_cf_objective(H_old,T,p));

x = H;
y = H;

t_init = 100;
tol = args.tolerance;
maxiter = args.max_iter;
iter = 0;
while stop == 0
    iter = iter + 1;
    if mod(iter,100) == 0
        display(['iteration = ' num2str(iter) ' Relative Error = ' num2str(rel_obj_err)]);
    end
    
    t = exact_line_search(y,T,p,@unconstrained_cf_gradient,@h_prox,args);
    t = backtracking_line_search(y,T,p,@unconstrained_cf_objective,@unconstrained_cf_gradient,@h_prox,args,t_init);
    [~,xplus] = h_prox(y - t*unconstrained_cf_gradient(y,T,p),args);
    yplus = xplus + ((iter-1)/(iter+2))*(xplus-x);
    obj_val = real(unconstrained_cf_objective(xplus,T,p));
    
    tlist = [tlist; t];
    obj_val_list = [obj_val_list;obj_val];
    rel_err = norm(H_old-xplus)/norm(H_old);
    rel_err_vect = [rel_err_vect, rel_err];
    
    rel_obj_err = abs(obj_val_list(end-1) - obj_val_list(end))/abs(obj_val_list(end-1));
    
    if (iter >= maxiter) || (rel_obj_err < tol)
        stop = 1;
        H = xplus;
    end
    H_old = xplus;
    x = xplus;
    y = yplus;
end

Hreshaped = reshape(H, fft_size);
hspatial = ifft2(Hreshaped,'symmetric')*fft_scale_factor;
hspatial = hspatial(1:args.img_size(1),1:args.img_size(2),:);

out.b = 0;
out.filt_freq = Hreshaped;
out.filt = hspatial;
out.args = args;
out.base = hspatial_base;

function [t,H] = backtracking_line_search(H,T,p,g_objective,g_gradient,h_prox,args,t_init)
beta = 0.5;  % (0 < beta < 1)
t = t_init;
f_H = real(g_objective(H,T,p));
grad = g_gradient(H,T,p);
dim = size(H,2);

stop = 0;
while stop == 0
    [~,Ht] = h_prox(H - t*grad,args);
    G = (1/t)*(H-Ht);
    f_Ht = real(g_objective(Ht,T,p));
    
    temp1 = real(sum(fusion_matrix_multiply(conj(grad),G,[1,dim],[dim,1])));
    temp2 = real(sum(fusion_matrix_multiply(conj(G),G,[1,dim],[dim,1])));
    
    if f_Ht > (f_H - t*temp1 + (t/2)*temp2)
        t = beta*t;
    else
        stop = 1;
        H = Ht;
    end
end

function t = exact_line_search(H,T,p,g_gradient,h_prox,args)

grad = g_gradient(H,T,p);
[~,grad] = h_prox(grad,args);

dim = size(H,2);
% Precompute some dots products
temp = fusion_matrix_multiply(T,H,[dim,dim],[dim,1]);
wd = sum(fusion_matrix_multiply(conj(grad),temp,[1,dim],[dim,1]));

temp = fusion_matrix_multiply(T,grad,[dim,dim],[dim,1]);
dd = sum(fusion_matrix_multiply(conj(grad),temp,[1,dim],[dim,1]));

wd = real(wd);
dd = real(dd);
dp = real(grad(:)'*p(:));
t = (wd-dp)/dd;