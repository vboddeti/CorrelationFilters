% compute_pce_plane.m
%	* This function measures the peak sharpness of the correlation output.
%   * It computes a metric known as peak-to-correlation energy ratio.
%	* When comparing scores across image scales you will need calibration. 
%   *
%	* Created by Vishnu Naresh Boddeti on 5/22/13.
%	* naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
%	* Copyright 2013 Carnegie Mellon University. All rights reserved.

function [pce,x,y] = compute_pce_plane(corrplane)

corrplane = normalize_image(corrplane);
pce = corrplane/std(corrplane(:));
[y,x] = find(abs(pce)==max(max(abs(pce))));