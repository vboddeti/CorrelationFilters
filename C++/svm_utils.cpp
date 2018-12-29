/*
 *  svm_utils.cpp
 *  CorrelationFilters
 *
 *  Modified by Vishnu Boddeti on 5/23/13.
 *	naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
 *  Copyright 2013 Carnegie Mellon University. All rights reserved.
 *
 *  TBD: There seems to be a memory leak, need to take a closer look.
 */

/*
 Author: Andrej Karpathy (http://cs.stanford.edu/~karpathy/)
 1 May 2012
 BSD licence
 */

#include "svm_utils.h"
#include "eigen_extensions.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

namespace esvm {
	
	SVMClassifier::SVMClassifier() {
		// Initialize parameters (defaults taken from libsvm code)
		param_= new svm_parameter();
		param_->svm_type = C_SVC;
		param_->kernel_type = LINEAR; //RBF;
		param_->degree = 3;
		param_->gamma = 0;	// 1/num_features
		param_->coef0 = 0;
		param_->nu = 0.5;
		param_->cache_size = 100;
		param_->C = 1.0;
		param_->eps = 1e-3;
		param_->p = 0.1;
		param_->shrinking = 1;
		param_->probability = 0;
		param_->nr_weight = 0;
		param_->weight_label = new int[1];
		param_->weight = new double[1];
		param_->weight_label[0] = 1;
		param_->weight[0] = 1.0;
		
		model_= NULL;
		problem_= NULL;
		x_space_= NULL;
		
		D_= 0;
	}
	
	SVMClassifier::~SVMClassifier() {
		
		svm_destroy_param(param_); //frees weight, weight_label
		free(param_);
		if(x_space_ != NULL) free(x_space_);
		
		// free model, if it exists
		if(model_ != NULL) svm_free_model_content(model_);
	}
	
	void SVMClassifier::train(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y) {
		
		// convert to vector and call real train
		int N= y.size();
		vector<int> yc(N, 0);
		for(int i=0;i<N;i++) { yc[i]= (int) y(i); }
		
		train(X, yc);
	}
	
	void SVMClassifier::train(const Eigen::MatrixXd &X, const vector<int> &y) {
		
		// Do some simple error checking before we begin
		int Ntr= X.rows();
		int D= X.cols();
		if(Ntr!=y.size()) {printf("ERROR: Number of labels should match number of examples! Exitting...\n"); return;}
		if(X.rows() == 0) {printf("No data provided in X: #rows = 0. Exitting...\n"); return;}
		if(X.cols() == 0) {printf("No data provided in X: #cols = 0. Exitting...\n"); return;}
		
		// Clean up memory allocated in the previous calls to train
		if(model_ != NULL) {
			svm_free_model_content(model_);
			model_= NULL;
		}
		
		if(x_space_ != NULL) {
			free(x_space_);
			x_space_ = NULL;
		}
		
		// Preallocate structures for transformation
		problem_ = new svm_problem();
		problem_->l= Ntr;
		problem_->y= Malloc(double, Ntr);
		problem_->x= Malloc(struct svm_node*, Ntr);
		x_space_= Malloc(struct svm_node, Ntr * (D+1)); // D+1 because of libsvm packing conventions
		D_= D; //store dimension of data
		
		// Process data from X matrix appropriately
		int j=0;
		for(int i=0;i<Ntr;i++) {
			problem_->x[i] = &x_space_[j];
			
			int k=0;
			while(k < D){
				
				x_space_[j].index= k+1; //features start at 1
				x_space_[j].value= X(i, k);
				
				k++;
				j++;
			}
			
			x_space_[j].index= -1; // Dark magic of inserting a -1 after every instance. Not sure what this is
			j++;
		}
		
		// Copy over data from label vector y
		for(int i=0;i<Ntr;i++) { problem_->y[i]= (double) y[i]; }
		
		// Make sure everything went ok in conversion
		const char *error_msg;
		error_msg = svm_check_parameter(problem_, param_);
		if(error_msg) {
			printf("ERROR in validating conversion: %s\n", error_msg);
		} else {
			// Train model!
			model_ = svm_train(problem_, param_);
		}
		
		// free memory for the problem data
		free(problem_->y);
		free(problem_->x);
	}
	
	void SVMClassifier::test(const Eigen::MatrixXd &X, vector<int> &yhat) {
		
		// Do simple error checking
		int Nte= X.rows();
		int D= X.cols();
		if(model_ == NULL) { printf("Error! Train an SVM first! Exitting... \n"); return;}
		
		// Carry out the classification
		yhat.resize(Nte);
		struct svm_node *x = Malloc(struct svm_node, D);
		double prob_estimates[2]; // assumes 2 classes
		
		for(int i=0;i<Nte;i++) {  
			
			for(int j=0;j<D;j++) {
				x[j].index = j+1;
				x[j].value = X(i, j);
			}
			
			double predict_label;
			if (param_->probability==1) {
				// prob_estimates here stores probability of class 1 and -1
				// eventually may want to have an option to accumulate and return these
				predict_label = svm_predict_probability(model_, x, prob_estimates);
			} else {
				predict_label= svm_predict_values(model_, x, prob_estimates);
			}
			yhat[i]= (int) predict_label;
		}
		
		free(x);
	}
	
	int SVMClassifier::saveModel(const char *filename) {
		int result= svm_save_model(filename, model_);
		return result;
	}
	
	void SVMClassifier::loadModel(const char *filename) {
		if(model_ != NULL) svm_free_model_content(model_); // deallocate first
		model_= svm_load_model(filename);
	}
	
	void SVMClassifier::setC(float Cnew) {
		param_->C= Cnew;
	}
	
	void SVMClassifier::setKernel(float Knew) {
		param_->kernel_type= Knew;
	}
	
	void SVMClassifier::setWpos(float Wnew) {
		param_->weight[0]= Wnew;
	}
	
	void SVMClassifier::getW(Eigen::MatrixXd &w, float &b) {
		if(model_ == NULL) { printf("Error! Train an SVM first! Exitting... \n"); return;}
		
		b= model_->rho[0]; //libsvm stores -b in rho
		
		// w is just linear combination of the support vectors when kernel is linear
		w.resize(D_, 1);
		for(int j=0;j<D_;j++) { 
			float acc=0;
			for(int i=0;i<model_->l;i++) {  
				acc += model_->SV[i][j].value * model_->sv_coef[0][i];
			}
			w(j)= -acc;
		}
	}
	
	void SVMClassifier::getNSV(int *nSV) {
		if(model_ == NULL) { printf("Error! Train an SVM first! Exitting... \n"); return;}
		*nSV= model_->l;
	}
	
	void SVMClassifier::getSI(Eigen::VectorXi &sv_indices) {
		if(model_ == NULL) { printf("Error! Train an SVM first! Exitting... \n"); return;}
		for (int k=0; k<model_->l; k++) {
			sv_indices[k] = model_->sv_indices[k];
		}
	}
	
	void SVMClassifier::getCoeff(Eigen::VectorXd &sv_coeff) {
		if(model_ == NULL) { printf("Error! Train an SVM first! Exitting... \n"); return;}
		for (int i=0; i<model_->l; i++) {
			sv_coeff[i] = model_->sv_coef[0][i];
		}
	}
};