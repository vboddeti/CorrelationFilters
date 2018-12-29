% build_svm_primal.m;
%
% * Citation: This function trains an SVM in the primal.
% * Created by Vishnu Naresh Boddeti on 5/22/13.
% * naresh@cmu.edu (http://vishnu.boddeti.net)
% * Copyright 2013 Carnegie Mellon University. All rights reserved.

function out = build_svm_primal

global args;
global data;
global labels;

if ~isfield(args,'cg'),                args.cg = 0;                        end;
if ~isfield(args,'lin_cg'),            args.lin_cg = 0;                    end;
if ~isfield(args,'iter_max_Newton'),   args.iter_max_Newton = 40;          end;
if ~isfield(args,'prec'),              args.prec = 1e-6;                   end;
if ~isfield(args,'cg_prec'),           args.cg_prec = 1e-4;                end;
if ~isfield(args,'cg_it'),             args.cg_it = 20;                    end;

lambda = 1/args.C;
img_size = size(data);
data = reshape(data,[prod(args.img_size) img_size(4)])';

% Call the right function depending on problem type and CG / Newton
% Also check that data / K exists and that the dimension of labels is correct
[n,d] = size(data);
if issparse(data), args.lin_cg = 1; end;
if size(labels,1)~=n, error('Dimension error'); end;
if ~args.cg
    [sol,obj] = primal_svm_linear(labels,lambda);
else
    [sol,obj] = primal_svm_linear_cg(labels,lambda);
end;

% The last component of the solution is the bias b.
b = sol(end);
sol = sol(1:end-1);
fprintf('\n');

out.filt = reshape(sol,args.img_size);
out.b = b;
data = reshape(data',img_size);

function  [w,obj] = primal_svm_linear(labels,lambda)
% -------------------------------
% Train a linear SVM using Newton
% -------------------------------
global args;
global data;
[n,d] = size(data);

w = zeros(d+1,1); % The last component of w is b.
iter = 0;
out = ones(n,1); % Vector containing 1-labels.*(data*w)

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
        Xsv = data(sv,:);
        hess = lambda*diag([ones(d,1); 0]) + ...   % Hessian
            [[Xsv'*Xsv sum(Xsv,1)']; [sum(Xsv) length(sv)]];
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

function  [w, obj] = primal_svm_linear_cg(labels,lambda)
% -----------------------------------------------------
% Train a linear SVM using nonlinear conjugate gradient
% -----------------------------------------------------
global args;
global data;
[n,d] = size(data);

w = zeros(d+1,1); % The last component of w is b.
iter = 0;
out = ones(n,1); % Vector containing 1-labels.*(data*w)
go = [data'*labels; sum(labels)];  % -gradient at w=0

s = go; % The first search direction is given by the gradient
while 1
    iter = iter + 1;
    if iter > args.cg_it * min(n,d)
        warning(sprintf(['Maximum number of CG iterations reached. ' ...
            'Try larger lambda']));
        break;
    end;
    
    % Do an exact line search
    [t,out] = line_search_linear(w,s,out,labels,lambda);
    w = w + t*s;
    
    % Compute the new gradient
    [obj, gn] = obj_fun_linear(w,labels,lambda,out); gn=-gn;
%     fprintf('Iter = %d, Obj = %f, Norm of grad = %.3f     \n',iter,obj,norm(gn));
    
    % Stop when the relative decrease in the objective function is small
    if t*s'*go < args.prec*obj, break; end;
    
    % Flecher-Reeves update. Change 0 in 1 for Polack-Ribiere
    be = (gn'*gn - 0*gn'*go) / (go'*go);
    s = be*s+gn;
    go = gn;
    
    fprintf('Iter = %d, Obj = %f, Norm of grad = %.3f  %.3f %.3f   \n',iter,obj,norm(gn),be,norm(s));
    
end;

function [obj, grad, sv] = obj_fun_linear(w,labels,lambda,out)
% Compute the objective function, its gradient and the set of support vectors
% Out is supposed to contain 1-labels.*(data*w)
global data
out = max(0,out);
w0 = w; w0(end) = 0;  % Do not penalize b
obj = sum(out.^2)/2 + lambda*w0'*w0/2; % L2 penalization of the errors
grad = lambda*w0 - [((out.*labels)'*data)'; sum(out.*labels)]; % Gradient
sv = find(out>0);

function labels = hess_vect_mult(w,sv,lambda)
% Compute the Hessian times a given vector data.
% hess = lambda*diag([ones(d-1,1); 0]) + (data(sv,:)'*data(sv,:));
global data
labels = lambda*w;
labels(end) = 0;
z = (data*w(1:end-1)+w(end));  % Computing data(sv,:)*data takes more time in Matlab :-(
zz = zeros(length(z),1);
zz(sv)=z(sv);
labels = labels + [(zz'*data)'; sum(zz)];

function [t,out] = line_search_linear(w,d,out,labels,lambda)
% From the current solution w, do a line search in the direction d by
% 1D Newton minimization
global data
t = 0;
% Precompute some dots products
Xd = data*d(1:end-1)+d(end);
wd = lambda * w(1:end-1)'*d(1:end-1);
dd = lambda * d(1:end-1)'*d(1:end-1);
while 1
    out2 = out - t*(labels.*Xd); % The new outputs after a step of length t
    sv = find(out2>0);
    g = wd + t*dd - (out2(sv).*labels(sv))'*Xd(sv); % The gradient (along the line)
    h = dd + Xd(sv)'*Xd(sv); % The second derivative (along the line)
    t = t - g/h; % Take the 1D Newton step. Note that if d was an exact Newton
    % direction, t is 1 after the first iteration.
    if g^2/h < 1e-10, break; end;
    %    fprintf('%f %f\n',t,g^2/h)
end;
out = out2;