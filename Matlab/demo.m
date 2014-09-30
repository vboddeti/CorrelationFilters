% demo.m

clear all
close all
clc

global data
global data_freq;
global labels;
global args;

im = double(imread('data/auth1.pgm'));
im = normalize_image(im);
[l1,l2] = gradient(im);
Y(:,:,1,1) = l1;
Y(:,:,2,1) = l2;
X(:,:,1,1) = im;

im = double(imread('data/auth2.pgm'));
im = normalize_image(im);
[l1,l2] = gradient(im);
Y(:,:,1,2) = l1;
Y(:,:,2,2) = l2;
X(:,:,1,2) = im;

im = double(imread('data/auth3.pgm'));
im = normalize_image(im);
[l1,l2] = gradient(im);
Y(:,:,1,3) = l1;
Y(:,:,2,3) = l2;
X(:,:,1,3) = im;

im = double(imread('data/imp1.pgm'));
im = normalize_image(im);
[l1,l2] = gradient(im);
Y(:,:,1,4) = l1;
Y(:,:,2,4) = l2;
X(:,:,1,4) = im;

im = double(imread('data/imp2.pgm'));
im = normalize_image(im);
[l1,l2] = gradient(im);
Y(:,:,1,5) = l1;
Y(:,:,2,5) = l2;
X(:,:,1,5) = im;

im = double(imread('data/imp3.pgm'));
im = normalize_image(im);
[l1,l2] = gradient(im);
Y(:,:,1,6) = l1;
Y(:,:,2,6) = l2;
X(:,:,1,6) = im;
labels = [1,1,1,-1,-1,-1];
labels = labels';

[labels,ind] = sort(labels,'descend');
X = X(:,:,:,ind);
Y = Y(:,:,:,ind);

%% Scalar Features
% X contains scalar features

[m,n,dim,num_img] = size(X);

args.C = 1;
args.alpha = 1e-3;
args.beta = 1-args.alpha;
args.img_size = [m,n,dim];
args.dim = dim;
args.t_init = 100;
args.tolerance = 1e-8;
args.max_iter = 1e6;
args.batch_size = 2500;
args.psd_flag = 0;
args.fft_size = [m,n];
args.fft_mask = ones(args.fft_size(1),args.fft_size(2),dim);
args.fft_mask(1:m,1:n,:) = 0;
args.target_magnitude = 1;
args.target_sigma = 0.2;
data = X;

data_freq = fft2(data,args.fft_size(1),args.fft_size(2))/sqrt(prod(args.fft_size));
otsdf = build_otsdf;
uotsdf = build_uotsdf;
% mmcf_dual = build_mmcf_dual;
mmcf_primal = build_mmcf_primal;

args.psd_flag = 0;
args.fft_size = 2*[m,n]-1;
args.fft_mask = ones(args.fft_size(1),args.fft_size(2),dim);
args.fft_mask(1:m,1:n,:) = 0;
data_freq = fft2(data,args.fft_size(1),args.fft_size(2))/sqrt(prod(args.fft_size));
zauotsdf = build_zauotsdf;
zammcf = build_zammcf_primal;

b = otsdf.b;
w = otsdf.filt;
for i = 1:num_img
    corrplane = 0;
    for j = 1:size(w,3)
        corrplane = corrplane + imfilter(X(:,:,j,i),w(:,:,j));
    end
    corrplane = corrplane + b;
    [score, loc, corrplane] = compute_pce_plane(corrplane);
    mesh(corrplane);title(score);pause(1);    
end

b = uotsdf.b;
w = uotsdf.filt;
for i = 1:num_img
    corrplane = 0;
    for j = 1:size(w,3)
        corrplane = corrplane + imfilter(X(:,:,j,i),w(:,:,j));
    end
    corrplane = corrplane + b;
    [score, loc, corrplane] = compute_pce_plane(corrplane);
    mesh(corrplane);title(score);pause(1);    
end

b = mmcf_primal.b;
w = mmcf_primal.filt;
for i = 1:num_img
    corrplane = 0;
    for j = 1:size(w,3)
        corrplane = corrplane + imfilter(X(:,:,j,i),w(:,:,j));
    end
    corrplane = corrplane + b;
    [score, loc, corrplane] = compute_pce_plane(corrplane);
    mesh(corrplane);title(score);pause(1);    
end

b = zauotsdf.b;
w = zauotsdf.filt;
for i = 1:num_img
    corrplane = 0;
    for j = 1:size(w,3)
        corrplane = corrplane + imfilter(X(:,:,j,i),w(:,:,j));
    end
    corrplane = corrplane + b;
    [score, loc, corrplane] = compute_pce_plane(corrplane);
    mesh(corrplane);title(score);pause(1);    
end

b = zammcf.b;
w = zammcf.filt;
for i = 1:num_img
    corrplane = 0;
    for j = 1:size(w,3)
        corrplane = corrplane + imfilter(X(:,:,j,i),w(:,:,j));
    end
    corrplane = corrplane + b;
    [score, loc, corrplane] = compute_pce_plane(corrplane);
    mesh(corrplane);title(score);pause(1);    
end

%% Vector Features
% Y contains vector features

[m,n,dim,num_img] = size(Y);

args.C = 1;
args.alpha = 1e-3;
args.beta = 1-args.alpha;
args.img_size = [m,n,dim];
args.dim = dim;
args.t_init = 100;
args.tolerance = 1e-8;
args.max_iter = 1e6;
args.batch_size = 2500;
args.psd_flag = 0;
args.fft_size = [m,n];
args.fft_mask = ones(args.fft_size(1),args.fft_size(2),dim);
args.fft_mask(1:m,1:n,:) = 0;
args.target_magnitude = 1;
args.target_sigma = 0.2;
data = Y;

data_freq = fft2(data,args.fft_size(1),args.fft_size(2))/sqrt(prod(args.fft_size));
otsdf = build_otsdf;
uotsdf = build_uotsdf;
% mmcf_dual = build_mmcf_dual;
mmcf_primal = build_mmcf_primal;

args.psd_flag = 0;
args.fft_size = 2*[m,n]-1;
args.fft_mask = ones(args.fft_size(1),args.fft_size(2),dim);
args.fft_mask(1:m,1:n,:) = 0;
data_freq = fft2(data,args.fft_size(1),args.fft_size(2))/sqrt(prod(args.fft_size));
zauotsdf = build_zauotsdf;
zammcf = build_zammcf_primal;

b = otsdf.b;
w = otsdf.filt;
for i = 1:num_img
    corrplane = 0;
    for j = 1:size(w,3)
        corrplane = corrplane + imfilter(Y(:,:,j,i),w(:,:,j));
    end
    corrplane = corrplane + b;
    [score, loc, corrplane] = compute_pce_plane(corrplane);
    mesh(corrplane);title(score);pause(1);    
end

b = uotsdf.b;
w = uotsdf.filt;
for i = 1:num_img
    corrplane = 0;
    for j = 1:size(w,3)
        corrplane = corrplane + imfilter(Y(:,:,j,i),w(:,:,j));
    end
    corrplane = corrplane + b;
    [score, loc, corrplane] = compute_pce_plane(corrplane);
    mesh(corrplane);title(score);pause(1);    
end

b = mmcf_primal.b;
w = mmcf_primal.filt;
for i = 1:num_img
    corrplane = 0;
    for j = 1:size(w,3)
        corrplane = corrplane + imfilter(Y(:,:,j,i),w(:,:,j));
    end
    corrplane = corrplane + b;
    [score, loc, corrplane] = compute_pce_plane(corrplane);
    mesh(corrplane);title(score);pause(1);    
end

b = zauotsdf.b;
w = zauotsdf.filt;
for i = 1:num_img
    corrplane = 0;
    for j = 1:size(w,3)
        corrplane = corrplane + imfilter(Y(:,:,j,i),w(:,:,j));
    end
    corrplane = corrplane + b;
    [score, loc, corrplane] = compute_pce_plane(corrplane);
    mesh(corrplane);title(score);pause(1);    
end

b = zammcf.b;
w = zammcf.filt;
for i = 1:num_img
    corrplane = 0;
    for j = 1:size(w,3)
        corrplane = corrplane + imfilter(Y(:,:,j,i),w(:,:,j));
    end
    corrplane = corrplane + b;
    [score, loc, corrplane] = compute_pce_plane(corrplane);
    mesh(corrplane);title(score);pause(1);    
end
