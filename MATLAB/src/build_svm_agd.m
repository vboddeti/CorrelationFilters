% build_svm_agd.m;

function out = build_svm_agd

global args;
global data;
global labels;

if ~isfield(args,'prec')
    args.prec = 1e-6;
end

lambda = 1/args.C;

img_size = size(data);
data = reshape(data,[prod(args.img_size) img_size(4)])';
[n,d] = size(data);
w = zeros(d+1,1); % The last component of w is b.
out = ones(n,1); % Vector containing 1-labels.*(data*w)
grad = [data'*labels; sum(labels)];

%% ZACF Proximal Gradient Descent Iterations

x = w;
y = w;
w_old = w;

obj_val_list = [];
iter = 0;
stop = 0;
while stop == 0
    iter = iter + 1;
    [t,out] = line_search_linear(w_old,grad,out,labels,lambda);
    [obj, grad, sv] = obj_fun_linear(w,labels,lambda,out);
    xplus = y+t*grad;
    yplus = xplus + ((iter-1)/(iter+2))*(xplus-x);
    yplus = xplus;
    
    obj_val_list = [obj_val_list;obj];
    if grad'*grad < args.prec*obj
        % Stop when the Newton decrement is small enough
        stop = 1;
        w = xplus;
        if iter == 400
            disp('');
        end
    end;
    fprintf('Iter = %d, Obj = %f, Norm of grad = %.3f     \n',iter,obj,norm(grad));
    w_old = xplus;
    x = xplus;
    y = yplus;
end

out.b = 0;
out.filt = reshape(w,args.img_size);
out.args = args;
data = reshape(data',img_size);


function [obj, grad, sv] = obj_fun_linear(w,labels,lambda,out)
% Compute the objective function, its gradient and the set of support vectors
% Out is supposed to contain 1-labels.*(data*w)
global data
out = max(0,out);
w0 = w; w0(end) = 0;  % Do not penalize b
obj = sum(out.^2)/2 + lambda*w0'*w0/2; % L2 penalization of the errors
grad = lambda*w0 - [((out.*labels)'*data)'; sum(out.*labels)]; % Gradient
sv = find(out>0);

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