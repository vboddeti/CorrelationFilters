/*
 *  utils.h
 *  CorrelationFilters
 *
 *  Created by Vishnu Boddeti on 7/1/13.
 *	naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
 *  Copyright 2013 Carnegie Mellon University. All rights reserved.
 *
 */

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"
#include <dirent.h>

//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>

void myunion(std::vector<int> &v, std::vector<int> &A, std::vector<int> &B);
void mysetdiff(std::vector<int> &v, std::vector<int> &A, std::vector<int> &B);