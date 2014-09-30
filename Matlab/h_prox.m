% h_prox.m
%
% * Created by Vishnu Naresh Boddeti on 9/30/14.
% * naresh@cmu.edu (http://vishnu.boddeti.net)
% * Copyright 2014 Carnegie Mellon University. All rights reserved.

function [H,H_vec,hspatial] = h_prox(H,opts)

init_size = size(H);
dim = opts.dim;
fft_size = opts.fft_size;
fft_scale_factor = sqrt(prod(fft_size(1:2)));
Hreshaped = reshape(H, [fft_size dim]);
hspatial = ifft2(Hreshaped,'symmetric')*fft_scale_factor;
hspatial(opts.fft_mask==1) = 0;
H = fft2(hspatial,opts.fft_size(1),opts.fft_size(2))/fft_scale_factor;
H_vec = reshape(H,[prod(fft_size(1:2)) dim]);

H = reshape(H,init_size);
H_vec = reshape(H_vec,init_size);