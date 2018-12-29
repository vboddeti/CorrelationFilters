% normalize_image.m
%
% * Created by Vishnu Naresh Boddeti on 9/30/14.
% * naresh@cmu.edu (http://vishnu.boddeti.net)
% * Copyright 2014 Carnegie Mellon University. All rights reserved.

function im = normalize_image(im)

im = double(im);
im = im - mean2(im);
im = im/sqrt(sum(sum(abs(im).^2)));