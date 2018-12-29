% build_zammcf_online.m

function out = build_zammcf_online

% Usage: [W,b] = pegasos(X,Y,lambda,k,maxIter)
% Input:
% X: n*d matrix, n=number of examples(instances), d=number of variables (features);
% Y: n*1 vector indicating class labels (1 or -1) for examples in X;
% lambda, k: parameters in Pegasos algorithm (default: lambda=1, k=0.1*n);
% maxIter: maximum number of iterations for W-vector convergence; (default: 10000);
% Tolerance: Allowable tolarance for norm of the differance between W-vectors in
% consecutive iterations, to be used as stopping criterion (default: 10^-6);
%
% Training stops if either of maxIter or Tolerance condition is satisfied;
%
% Output:
% W,b: parameters in SVM primal problem:
%
% min 0.5*(||W||)^2
% s.t. (W'*Xi+b)*yi >= 1, for all i=1,...,n
% yi={1,-1};
%
% This function is implementation of Pegasos paper for SVM classification problem.
% Paper referance:
% "Pegasos-Primal Estimated sub-Gradient SOlver for SVM"
% By Shwartz, Singer and Srebro : 2007
%
% Code by:
% Vishnu Naresh Boddeti

global D;
global args;
global labels;
global data;
global data_freq

[m,n,dim,num] = size(data_freq);

if ~args.psd_flag
    compute_psd;
    args.psd_flag = 1;
end

alpha = args.alpha;
beta = args.beta;
[d,dim] = size(D);
ind = 1:dim:dim^2;
S = beta*D;
S(:,ind) = alpha*ones(d,dim)+S(:,ind);

lambda = 1/(args.C*num);
k1 = min(0.2*num,100);
maxIter = args.max_iter;
tol = args.tolerance;

pos_idx = find(labels==1);
neg_idx = find(labels==-1);
num_pos = length(pos_idx);
num_neg = length(neg_idx);

data_freq = reshape(data_freq,[m*n*dim,num]);
w = rand([m,n,dim]);
w_freq = fft2(w)/sqrt(m*n);
w_freq = reshape(w_freq,[m*n*dim,1]);
w_avg_freq = w_freq;
err = zeros(maxIter,1);
for t = 1:maxIter
    %     fprintf('\niteration # %d/%d',t,maxIter);
    w_old_freq = w_freq;
    
    b_freq = mean(labels - data_freq'*w_freq);
    idx = randi(num,[k1,1]);
    if num_pos/num_neg < 0.4
        idx = [idx;pos_idx];
    end
    k = length(idx);
    At = data_freq(:,idx);
    yt = labels(idx);
    idx1 = (At'*w_freq+b_freq).*yt<1;
    etat = 1/(lambda*t);
    
    tmp = reshape(w_freq,[m*n,dim]);
    tmp = fusion_matrix_multiply(S,tmp,[dim,dim],[dim,1]);
    tmp = reshape(tmp,[m*n*dim,1]);
    
    w1 = w_freq - etat*lambda*tmp + (etat/k)*sum(At(:,idx1).*repmat(yt(idx1)',[size(At,1),1]),2);
    
    w1 = reshape(w1,[m,n,dim]);
    w1 = h_prox(w1,args);
    w1 = reshape(w1,[m*n*dim,1]);
    
    w_freq = min(1,1/(sqrt(lambda)*norm(w1)))*w1;
    w_avg_freq = w_avg_freq + w_freq;
    
    err(t,1) = norm(w_freq-w_old_freq)/norm(w_old_freq);
    %     fprintf(' err = %f', tmp);
    if(err(t,1) < tol)
        break;
    end
end

err = err(1:t,1);
data_freq = reshape(data_freq,[m,n,dim,num]);
if(t<maxIter)
    fprintf('\nW converged in %d iterations.',t);
else
    fprintf('\nW not converged in %d iterations.',maxIter);
end
w_avg_freq = w_avg_freq/max(1,t);
w_avg_freq = reshape(w_avg_freq,[m,n,dim]);
w1 = ifft2(w_avg_freq,'symmetric')*sqrt(m*n);

[m,n,dim,num] = size(data);
w1 = w1(1:m,1:n,:);
data = reshape(data,[m*n*dim,num]);
w1 = reshape(w1,[m*n*dim,1]);

b = mean(labels-data'*w1);
scores = data'*w1+b;
[X,Y,T,AUC] = perfcurve(labels,scores,1);
fprintf('\n Online ZAMMCF: Pegasos Accuracy on Training set = %.4f %%\n', 100*AUC);

data = reshape(data,[m,n,dim,num]);
out.filt = reshape(w1,[m,n,dim]);
out.b = b;
out.args = args;