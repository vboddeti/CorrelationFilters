/*
 *  svm_utils.h
 *  CorrelationFilters
 *
 *  Modified by Vishnu Boddeti on 5/23/13.
 *	naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
 *  Copyright 2013 Carnegie Mellon University. All rights reserved.
 *
 */

/*
 Author: Andrej Karpathy (http://cs.stanford.edu/~karpathy/)
 1 May 2012
 BSD licence
 */

#ifndef __EIGEN_SVM_UTILS_H__
#define __EIGEN_SVM_UTILS_H__

#include <string>
#include <vector>
#include "svm.h"
#include <eigen3/Eigen/Eigen>

using namespace std;
namespace esvm {
	
	/* 
	 Trains a binary SVM on dense data with linear kernel using libsvm. 
	 Usage:
	 
	 vector<int> yhat;
	 SVMClassifier svm;
	 svm.train(X, y);
	 svm.test(X, yhat);
	 
	 where X is an Eigen::MatrixXf that is NxD array. (N D-dimensional data),
	 y is a vector<int> or an Eigen::MatrixXf Nx1 vector. The labels are assumed
	 to be -1 and 1. This version doesn't play nice if your dataset is 
	 too unbalanced.
	 */
	class SVMClassifier{
    public:
		
		SVMClassifier();
		~SVMClassifier();
		
		// train the svm
		void train(const Eigen::MatrixXd &X, const vector<int> &y);
		void train(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y);
		
		// test on new data 
		void test(const Eigen::MatrixXd &X, vector<int> &yhat);
		
		// libsvm does not directly calculate the w and b, but a set of support
		// vectors. This function will use them to compute w and b, as currenly
		// we assume linear kernel only
		// yhat = sign( X * w + b )
		void getW(Eigen::MatrixXd &w, float &b);
		void getNSV(int *nSV);
		void getSI(Eigen::VectorXi &sv_indices);
		void getCoeff(Eigen::VectorXd &sv_coeff);
		
		// I/O
		int saveModel(const char *filename);
		void loadModel(const char *filename); 
		
		// options
		void setC(float Cnew); //default is 1.0
		void setKernel(float Knew); //default is 0.0, linear kernel.
		void setWpos(float Wnew); //default is 1.0
		
		//TODO: add cross validation support
		//TODO: add probability support?
		
		svm_model *model_;
		
    protected:
		
		svm_problem *problem_;
		svm_parameter *param_;
		svm_node *x_space_;
		
		int D_; //dimension of data
	};
};

#endif //__EIGEN_SVM_UTILS_H__