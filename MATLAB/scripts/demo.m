% demo.m

clear all
close all
clc

args.alpha = 1e-3;
args.beta = 1-args.alpha;
args.C = 1;
args.wpos = 1;

im = double(imread('../data/auth1.pgm'));
im = normalize_image(im);
[l1,l2] = gradient(im);
Y(:,:,1,1) = l1;
Y(:,:,2,1) = l2;
X(:,:,1,1) = im;

im = double(imread('../data/auth2.pgm'));
im = normalize_image(im);
[l1,l2] = gradient(im);
Y(:,:,1,2) = l1;
Y(:,:,2,2) = l2;
X(:,:,1,2) = im;

im = double(imread('../data/auth3.pgm'));
im = normalize_image(im);
[l1,l2] = gradient(im);
Y(:,:,1,3) = l1;
Y(:,:,2,3) = l2;
X(:,:,1,3) = im;

im = double(imread('../data/imp1.pgm'));
im = normalize_image(im);
[l1,l2] = gradient(im);
Y(:,:,1,4) = l1;
Y(:,:,2,4) = l2;
X(:,:,1,4) = im;

im = double(imread('../data/imp2.pgm'));
im = normalize_image(im);
[l1,l2] = gradient(im);
Y(:,:,1,5) = l1;
Y(:,:,2,5) = l2;
X(:,:,1,5) = im;

im = double(imread('../data/imp3.pgm'));
im = normalize_image(im);
[l1,l2] = gradient(im);
Y(:,:,1,6) = l1;
Y(:,:,2,6) = l2;
X(:,:,1,6) = im;
labels = [1,1,1,-1,-1,-1];

[labels,ind] = sort(labels,'descend');
X = X(:,:,:,ind);
Y = Y(:,:,:,ind);

%% Scalar Features

num_channels = 1;
args.size = [size(im) 1];
num_img = 6;

for i = 1:num_img
    img(i).im = X(:,:,:,i);
    img(i).label = labels(i);
end

mmcf_scalar = build_mmcf_scalar(img,args);
otsdf_scalar = build_otsdf_scalar(img,args);
uotsdf_scalar = build_uotsdf_scalar(img,args);

for i = 1:num_img
    corrplane = xcorr2(X(:,:,1,i),mmcf_scalar.filt);
    [corrplane,y,x] = compute_pce_plane(corrplane);
    score = corrplane(x,y);
    mesh(corrplane);title(score);pause(1);
end

for i = 1:num_img
    corrplane = xcorr2(X(:,:,1,i),otsdf_scalar.filt);
    [corrplane,y,x] = compute_pce_plane(corrplane);
    score = corrplane(x,y);
    mesh(corrplane);title(score);pause(1);
end

for i = 1:num_img
    corrplane = xcorr2(X(:,:,1,i),uotsdf_scalar.filt);
    [corrplane,y,x] = compute_pce_plane(corrplane);
    score = corrplane(x,y);
    mesh(corrplane);title(score);pause(1);
end

%% Vector Features

num_channels = 2;
args.size = [size(im) 2];
num_img = 6;
for i = 1:num_img
    img(i).im = Y(:,:,:,i);
    img(i).label = labels(i);
end

mmcf_vector = build_mmcf_vector(img,args);
otsdf_vector = build_otsdf_vector(img,args);
uotsdf_vector = build_uotsdf_vector(img,args);

for i = 1:num_img
    corrplane = 0;
    for j = 1:num_channels
        corrplane = corrplane + xcorr2(Y(:,:,j,i),mmcf_vector.filt(:,:,j));
    end
    [corrplane,y,x] = compute_pce_plane(corrplane);
    score = corrplane(x,y);
    mesh(corrplane);title(score);pause(1);    
end

for i = 1:num_img
    corrplane = 0;
    for j = 1:num_channels
        corrplane = corrplane + xcorr2(Y(:,:,j,i),otsdf_vector.filt(:,:,j));
    end
    [corrplane,y,x] = compute_pce_plane(corrplane);
    score = corrplane(x,y);
    mesh(corrplane);title(score);pause(1);
end

for i = 1:num_img
    corrplane = 0;
    for j = 1:num_channels
        corrplane = corrplane + xcorr2(Y(:,:,j,i),uotsdf_vector.filt(:,:,j));
    end
    [corrplane,y,x] = compute_pce_plane(corrplane);
    score = corrplane(x,y);
    mesh(corrplane);title(score);pause(1);
end