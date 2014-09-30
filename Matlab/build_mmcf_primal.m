% build_mmcf_primal.m;
%
% * Citation: This function implements the correlation filter design proposed in the publications.
% * A. Rodriguez, Vishnu Naresh Boddeti, B.V.K. Vijaya Kumar and A. Mahalanobis, "Maximum Margin Correlation Filter: A New Approach for
% * Localization and Classification", IEEE Transactions on Image Processing, 2012.
% * Vishnu Naresh Boddeti and B.V.K. Vijaya Kumar, "Maximum Margin Vector Correlation Filter" Arxiv 1404.6031 (April 2014)
% * Vishnu Naresh Boddeti, "Advances in Correlation Filters: Vector Features, Structured Prediction and Shape Alignment" PhD thesis,
% * Carnegie Mellon University, Pittsburgh, PA, USA, 2012.
% * Notes: This currently the best performing Correlation Filter design, especially when the training sample size is
% * larger than the dimensionality of the data.
%
% * Created by Vishnu Naresh Boddeti on 9/30/14.
% * naresh@cmu.edu (http://vishnu.boddeti.net)
% * Copyright 2014 Carnegie Mellon University. All rights reserved.

function out = build_mmcf_primal

global p;
global S;
global D;
global T;
global args;
global labels;
global data_freq;

if ~isfield(args,'cg'),                args.cg = 1;                        end;
if ~isfield(args,'lin_cg'),            args.lin_cg = 0;                    end;
if ~isfield(args,'iter_max_Newton'),   args.iter_max_Newton = 40;          end;
if ~isfield(args,'prec'),              args.prec = 1e-8;                   end;
if ~isfield(args,'cg_prec'),           args.cg_prec = 1e-6;                end;
if ~isfield(args,'cg_it'),             args.cg_it = 40;                    end;
if ~isfield(args,'verbose'),           args.verbose = 0;                   end;

if ~args.psd_flag
    compute_psd;
    args.psd_flag = 1;
end

dim = args.dim;
d = prod(args.fft_size);
alpha = args.alpha;
beta = args.beta;
ind = 1:(dim+1):dim^2;
T = beta*D;
T(:,ind) = alpha*ones(d,dim)+T(:,ind);
p = compute_mean;
p = reshape(p,[prod(args.fft_size),dim]);

if ~args.cg
    % Careful, this creates a huge matrix, use for low dimensional data
    % only.
    index = 1;
    S = zeros(d*dim+1,d*dim+1);
    for i = 1:dim
        for j = 1:dim
            S((i-1)*d+1:i*d,(j-1)*d+1:j*d) = diag(T(:,index));
            index = index + 1;
        end
    end
end

lambda = 1/args.C;
fft_size = size(data_freq);
data_freq = transpose(reshape(data_freq,[prod(fft_size(1:3)) fft_size(4)]));

% Call the right function depending on problem type and CG / Newton
% Also check that data_freq / K exists and that the dimension of labels is correct
[n,d] = size(data_freq);
if issparse(data_freq), args.lin_cg = 1; end;
if size(labels,1)~=n, error('Dimension error'); end;
if ~args.cg
    [sol,obj] = primal_linear_newton(labels,lambda,args);
else
    [sol,obj] = primal_linear_cg(labels,lambda,args);
end;

% The last component of the solution is the bias b.
b = real(sol(end));
sol = sol(1:end-1);
fprintf('\n');

out.filt_freq = reshape(sol,[args.fft_size args.dim]);
hspatial = ifft2(out.filt_freq,'symmetric')*sqrt(prod(args.fft_size));
hspatial = hspatial(1:args.img_size(1),1:args.img_size(2),:);
out.filt = hspatial;
out.b = b;
data_freq = reshape(transpose(data_freq),fft_size);

function  [w,obj] = primal_linear_newton(labels,lambda,args)
% -------------------------------
% Train a linear MMCF using Newton
% -------------------------------
global S;
global data_freq;
[n,d] = size(data_freq);

w = zeros(d+1,1); % The last component of w is b.
iter = 0;
out = ones(n,1); % Vector containing 1-labels.*(data_freq*w)

while 1
    iter = iter + 1;
    if iter > args.iter_max_Newton;
        warning(sprintf(['Maximum number of Newton steps reached.' ...
            'Try larger lambda']));
        break;
    end;
    
    [obj, grad, sv] = obj_fun_linear(w,labels,lambda,out);
    
    % Compute the Newton direction either exactly or by linear CG
    if args.lin_cg
        % Advantage of linear CG when using sparse input: the Hessian is never
        %   computed explicitly.
        [step, foo, relres] = minres(@hess_vect_mult, -grad,...
            args.cg_prec,args.cg_it,[],[],[],sv,lambda);
    else
        Xsv = data_freq(sv,:);
        hess = lambda*S + ...   % Hessian
            [[Xsv'*Xsv transpose(sum(Xsv,1))]; [sum(Xsv) length(sv)]];
        step  = - hess \ grad;   % Newton direction
    end;
    
    % Do an exact line search
    [t,out] = line_search_linear(w,step,out,labels,lambda);
    
    w = w + t*step;
    fprintf(['Iter = %d, Obj = %f, Nb of sv = %d, Newton decr = %.3f, ' ...
        'Line search = %.3f'],iter,obj,length(sv),-step'*grad/2,t);
    if args.lin_cg
        fprintf(', Lin CG acc = %.4f     \n',relres);
    else
        fprintf('      \n');
    end;
    
    if -step'*grad < args.prec * obj
        % Stop when the Newton decrement is small enough
        break;
    end;
end;

function  [w, obj] = primal_linear_cg(labels,lambda,args)
% -----------------------------------------------------
% Train a linear MMCF using nonlinear conjugate gradient
% -----------------------------------------------------
global data_freq;
[n,d] = size(data_freq);

w = zeros(d+1,1); % The last component of w is b.
iter = 0;
out = ones(n,1); % Vector containing 1-labels.*(data_freq*w)
go = [transpose(data_freq)*labels; sum(labels)];  % -gradient at w=0

s = go; % The first search direction is given by the gradient
while 1
    iter = iter + 1;
    if iter > min(args.cg_it * min(n,d),args.max_iter)
        warning(sprintf(['Maximum number of CG iterations reached. ' ...
            'Try larger lambda']));
        break;
    end;
    
    % Do an exact line search
    [t,out] = line_search_linear(w,s,out,labels,lambda);
    w = w + t*s;
    
    % Compute the new gradient
    [obj, gn, sv] = obj_fun_linear(w,labels,lambda,out); gn=-gn;
    
    % Stop when the relative decrease in the objective function is small
    if t*real(s'*go) < args.prec*obj, break; end;
    
    % Flecher-Reeves update. Change 0 in 1 for Polack-Ribiere
    % For Polack-Ribiere, set be=max(0,be);
    be = (gn'*gn - 0*gn'*go) / (go'*go);
    
    % good idea to reset beta once in a while    
    if ~mod(iter,25)
        be = 0;
    end
    
    s = be*s+gn;
    go = gn;
    
    if args.verbose
        fprintf('Iter = %d, Obj = %f, Norm of grad = %.3f, Nb of sv = %d \n',iter,obj,norm(gn),length(sv));
    end
end;

function [obj, grad, sv] = obj_fun_linear(w,labels,lambda,out)
% Compute the objective function, its gradient and the set of support vectors
% Out is supposed to contain 1-labels.*(data_freq*w)
global data_freq
out = max(0,out);
w0 = w(1:end-1); w0(end) = 0;  % Do not penalize b

[val, grad1] = mmcf_objective(w0);
obj = sum(out.^2)/2 + lambda*val;
grad1 = [grad1;0];
grad = lambda*grad1 - [transpose((out.*labels)'*data_freq); sum(out.*labels)]; % Gradient
sv = find(out>0);

function labels = hess_vect_mult(w,sv,lambda)
% Compute the Hessian times a given vector data_freq.
% hess = lambda*diag([ones(d-1,1); 0]) + (data_freq(sv,:)'*data_freq(sv,:));
global data_freq
labels = lambda*w;
labels(end) = 0;
z = ((data_freq)*w(1:end-1)+w(end));  % Computing data_freq(sv,:)*data_freq takes more time in Matlab :-(
zz = zeros(length(z),1);
zz(sv)=z(sv);
labels = labels + [(zz'*data_freq)'; sum(zz)];

function [t,out] = line_search_linear(w,d,out,labels,lambda)
% From the current solution w, do a line search in the direction d by
% 1D Newton minimization
global T;
global args;
global data_freq;

t = 0;
dim = args.dim;
% Precompute some dots products
Xd = conj(data_freq)*d(1:end-1)+d(end);
w_temp = w(1:end-1);
d_temp = d(1:end-1);
w_temp = reshape(w_temp,[prod(args.fft_size) dim]);
d_temp = reshape(d_temp,[prod(args.fft_size) dim]);

temp = fusion_matrix_multiply(T,w_temp,[dim,dim],[dim,1]);
wd = sum(fusion_matrix_multiply(conj(d_temp),temp,[1,dim],[dim,1]));

temp = fusion_matrix_multiply(T,d_temp,[dim,dim],[dim,1]);
dd = sum(fusion_matrix_multiply(conj(d_temp),temp,[1,dim],[dim,1]));

wd = lambda*real(wd);
dd = lambda*dd;

while 1
    out2 = out - t*(labels.*Xd); % The new outputs after a step of length t
    sv = find(out2>0);
    g = wd + t*dd - (out2(sv).*labels(sv))'*Xd(sv); % The gradient (along the line)
    h = dd + Xd(sv)'*Xd(sv); % The second derivative (along the line)
    t = t - real(g/h); % Take the 1D Newton step. Note that if d was an exact Newton
    % direction, t is 1 after the first iteration.
    if real(g^2/h) < 1e-10, break; end;
    %    fprintf('%f %f\n',t,g^2/h)
end;
out = real(out2);

function [val, grad] = mmcf_objective(w)
global T;
global args;
w_size = size(w);
w = reshape(w,[prod(args.fft_size) args.dim]);
dim = sqrt(size(T,2));
grad = fusion_matrix_multiply(T,w,[dim,dim],[dim,1]);
val = sum(fusion_matrix_multiply(conj(w),grad,[1,dim],[dim,1]))/2;
val = real(val);
grad = reshape(grad,w_size);