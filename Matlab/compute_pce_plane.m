% compute_pce_plane.m
%
% * Created by Vishnu Naresh Boddeti on 9/30/14.
% * naresh@cmu.edu (http://vishnu.boddeti.net)
% * Copyright 2014 Carnegie Mellon University. All rights reserved.

function [score,loc, corrplane] = compute_pce_plane(corrplane, abs_flag)

if nargin == 1
    abs_flag = 1;
end

if sum(corrplane(:)) ~= 0
    corrplane = normalize_image(corrplane);
    corrplane = corrplane/std(corrplane(:));
end

if abs_flag
    [x,y] = find(abs(corrplane) == max(abs(corrplane(:))),1,'first');
else
    [x,y] = find(corrplane == max(corrplane(:)),1,'first');
end

loc = [x,y];
score = corrplane(x,y);