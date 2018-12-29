/*
 *  CCorrelationFilters.h
 *  CorrelationFilters
 *
 *  Created by Vishnu Boddeti on 5/22/13.
 *	naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
 *  Copyright 2013 Carnegie Mellon University. All rights reserved.
 *
 */

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <complex>
#include <fftw3.h>
#include <vector>
#include <algorithm>
#include "svm_utils.h"
#include "utils.h"

// Define data structure for images in the correlation filter library.
struct CDataStruct{
	double *data;                                   // pointer to all signals
	complex<double> *data_freq;						// pointer to all signals in frequency domain
	int num_data;									// number of signals total
	int num_channels;								// number of channels (not number of dimensions)
	std::vector<double*> ptr_data;					// vector of pointers to each individual signal in *data
	std::vector<complex <double> *> ptr_data_freq;	// vector of pointers to each individual signal in *data_freq
	Eigen::VectorXi size_data;						// number of elements will determine dimension of individual signal (1 number = 1d, 2 numbers = 2d, 3d, etd), each element will tell size of the signal (row, col, depth)
	Eigen::VectorXi size_data_freq;					// number of elements will determine dimension of individual signal in frequency domain (1 number = 1d, 2 numbers = 2d, 3d, etd), each element will tell size of the signal (row, col, depth)
	double *labels;									// +1 or -1 (svm uses doubles)
	int num_elements_freq;							// number of frequency components from an individual signal (if real signal then only 1/2 the number of the cardinality of the signal is needed)
    Eigen::ArrayXXcd D;
    Eigen::ArrayXXcd S;
    Eigen::ArrayXXcd Sinv;
    bool psd_flag;
    bool inv_psd_flag;
	
    CDataStruct()
    {
        data = NULL;
        data_freq = NULL;
        num_data = 0;
        num_channels = 0;
        labels = NULL;
        num_elements_freq = 0;
        size_data = Eigen::VectorXi::Zero(0);
        size_data_freq = Eigen::VectorXi::Zero(0);
        ptr_data.reserve(0);
        ptr_data_freq.reserve(0);
    }
    
//	~CDataStruct(){
//		delete data;								// clean up the image data
//		delete data_freq;							// clean up the image frequency data
//        delete labels;								// clean up the labels
//	}
};

// Define data structure for filter design params in the correlation filter library.
struct CParamStruct{
	double C;										// libsvm slack variable parameter 'C'
	double alpha;									// alpha = 1 = svm (regularization), alpha = 0 = ACE (no regularization)
	double beta;									// beta = 1-alpha
	int kernel_type; 								// Follows libsvm convention (only linear is supported)
	Eigen::VectorXi size_filt;						// number of elements will determine dimension of individual template (1 number = 1d, 2 numbers = 2d, 3d, etd), each element will tell size of the signal (row, col, depth)
	Eigen::VectorXi size_filt_freq;					// number of elements will determine dimension of individual template in frequency domain (1 number = 1d, 2 numbers = 2d, 3d, etd), each element will tell size of the signal (row, col, depth)
	double whiten_flag;								// if 0 = only positive signals, 1 = only negative signals, 2 = use all signals to whiten
	double neg_flag;								// if 0 = only positive signals, 1 = only negative signals, 2 = use all signals to compute mean on unconstrained filters
	double wpos;									// weight that is given to a positive sample (used in libsvm)
	int num_elements_freq;							// number of frequency components from an individual template (if real signal then only 1/2 the number of the cardinality of the signal is needed)
    
    CParamStruct()
    {
        C = 1;
        alpha = 1e-2;
        beta = 1-alpha;
        kernel_type = 0;
        whiten_flag = 2;
        neg_flag = 2;
        wpos = 1;
        size_filt = Eigen::VectorXi::Zero(0);
        size_filt_freq = Eigen::VectorXi::Zero(0);
        num_elements_freq = 0;
    }
};

// Define data structure for filters in the correlation filter library.
struct CFilterStruct{
	CDataStruct filter;
	CParamStruct params;
};

class CCorrelationFilters{
    
public:
    
	// zero mean, unit variance each signal
	void normalize_data(struct CDataStruct *img);
	
	// saves the filter structure to a binary file
	void save_filter(struct CFilterStruct *filt, const char *filename);
	
	// loads the filter structure from a binary file
	void load_filter(struct CFilterStruct *filt, const char *filename);
	
	// not implemented, if you have two filter structures with the same parameters then merge them into one structure
	void merge_filters(struct CFilterStruct *filt1, struct CFilterStruct *filt2);
	
	// corr = correlation outputs, img = data structure, filt = filter structure
	void apply_filter(struct CDataStruct *corr, struct CDataStruct *img, struct CFilterStruct *filt);
	
	// returns mmcf templates in the filter structure
	void build_mmcf(struct CDataStruct *img, struct CParamStruct *params, struct CFilterStruct *filt);
	
	// returns otdsdf templates in the filter structure
	void build_otsdf(struct CDataStruct *img, struct CParamStruct *params, struct CFilterStruct *filt);
	
	// returns uotsdf templates in the filter structure
	void build_uotsdf(struct CDataStruct *img, struct CParamStruct *params, struct CFilterStruct *filt);
	
    // populates the data structure given the input components
	void initialize_data(struct CDataStruct *img,double *data,double *labels,int num_img, int *data_size,int num_dim,int num_channels);
    
    // prepare and process the data
	void prepare_data(struct CDataStruct *img, struct CParamStruct *params);
    
//protected:
	
	// populates the frequency data from the real data within the structure
	void fft_data(struct CDataStruct *img);
	
	// populates the real data from the frequency data within the structure
	void ifft_data(struct CDataStruct *img);
	
	// 1d,2d,3d zero padding, adds zeros at the end so that the data matches the size of the frequency data in the structure
	void zero_pad_data(double *data, struct CDataStruct *img);
	
	// inverts a matrix with our special structures, matrix with diagonal blocks
	void fusion_matrix_inverse(Eigen::ArrayXXcd &X, Eigen::MatrixXi indices);
	
    // computes the cross power-spectral-density of the data
    void compute_psd_matrix(struct CDataStruct *img, struct CParamStruct *params);
    
    // computes the regularized power-spectral-density and its inverse
	void compute_inverse_psd_matrix(struct CDataStruct *img, struct CParamStruct *params);
	
	// computes the product of two matrices with our special structure
	void fusion_matrix_multiply(Eigen::ArrayXXcd &out, Eigen::ArrayXXcd &A, Eigen::ArrayXXcd &B, Eigen::Vector2i num_blocks_A, Eigen::Vector2i num_blocks_B);
};