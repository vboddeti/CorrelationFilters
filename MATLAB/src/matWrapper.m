% matWrapper.m
%
%	* Created by Yair Movshovitz-Attias on 8/22/13.
%	* yair@cs.cmu.edu (http://www.cs.cmu.edu/~ymovshov/)
%	* Copyright 2013 Carnegie Mellon University. All rights reserved.
% 	* Notes : This class wraps a full (single) matrix so that it can be passed by reference to functions.

classdef matWrapper < handle
   properties
      Data
   end
   methods
      function obj = matWrapper(shape)
         if nargin > 0
            obj.Data = zeros(shape);           
         end
      end
   end
end