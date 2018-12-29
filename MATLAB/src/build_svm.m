% build_svm.m

function out = build_svm

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

global args;
global labels;
global data;

[m,n,dim,num] = size(data);

data = reshape(data,[m*n*dim,num]);
K = data'*data;
K = double((K+K')/2);
K = [(1:num)' K];
options = ['-s 0 -t 4 -c ' num2str(args.C)];
mmcf = svmtrain(labels,K,options);
b = -mmcf.rho;
w = data(:,mmcf.sv_indices)*mmcf.sv_coef;
w = reshape(w,[m,n,dim]);

data = reshape(data,[m,n,dim,num]);
out.filt = reshape(w,[m,n,dim]);
out.b = b;
out.args = args;