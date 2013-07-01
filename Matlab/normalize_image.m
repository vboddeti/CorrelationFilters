% normalize_image.m
%
%	* Created by Vishnu Naresh Boddeti on 5/22/13.
%	* naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
%	* Copyright 2013 Carnegie Mellon University. All rights reserved.

function im = normalize_image(im)

im = double(im);
im = im - mean2(im);
im = im/sqrt(sum(sum(abs(im).^2)));