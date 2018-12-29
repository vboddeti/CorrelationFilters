/*
 *  CCorrelationFilters.cpp
 *  CorrelationFilters
 *
 *  Created by Vishnu Boddeti on 5/22/13.
 *	naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
 *  Copyright 2013 Carnegie Mellon University. All rights reserved.
 *
 *  Notes: This library implements most of the popular and currently best performing correlation filters.
 *  TBD: Implement OTCHF, MACE-MRH, and QCF since they may be useful in some applications.
 *  TBD: Get multithreaded versions of FFTW and Eigen working.
 */

#include "CCorrelationFilters.h"

void CCorrelationFilters::initialize_data(struct CDataStruct *img,double *data,double *labels,int num_img, int *data_size,int num_dim,int num_channels=1)
{
	img->data = data;
	img->labels = labels;
	img->num_data = num_img;
	img->num_channels = num_channels;
	
	img->size_data = Eigen::VectorXi::Zero(num_dim);
	img->size_data_freq = Eigen::VectorXi::Zero(num_dim);
	
	for (int i=0; i<num_dim; i++) {
		img->size_data(i) = data_size[i];
		img->size_data_freq(i) = data_size[i];
	}
	
	img->ptr_data.reserve(num_img);
	img->ptr_data_freq.reserve(num_img);
	
	for (int i=0;i<num_img;i++){
        img->ptr_data.push_back((data + i*img->size_data.prod()*img->num_channels));
	}
}

void CCorrelationFilters::prepare_data(struct CDataStruct *img, struct CParamStruct *params)
{
    normalize_data(img);
    img->size_data_freq = params->size_filt_freq;
    fft_data(img);
    params->num_elements_freq = img->num_elements_freq;
    compute_psd_matrix(img, params);
    compute_inverse_psd_matrix(img, params);
}

void CCorrelationFilters::normalize_data(struct CDataStruct *img)
{
	Eigen::Map<Eigen::ArrayXXd> temp(img->data,img->size_data.prod()*img->num_channels,img->num_data);
	for (int i=0; i<img->num_data; i++) {
		temp.block(0,i,img->size_data.prod()*img->num_channels,1) -= temp.block(0,i,img->size_data.prod()*img->num_channels,1).mean();
		temp.block(0,i,img->size_data.prod()*img->num_channels,1) /= temp.block(0,i,img->size_data.prod()*img->num_channels,1).matrix().norm();
	}
}

void CCorrelationFilters::zero_pad_data(double *data, struct CDataStruct *img)
{
	// If the FFT size is NOT the same as the size of the data, zero pad the data. 
	// There does not seem to be an elegant way to zero pad data of different dimensions,
	// without explicitly handling data with different dimensions, hence the ugliness that follows.
	
	int rank = img->size_data.size();
	int num = img->num_data*img->num_channels*img->size_data_freq.prod();
	assert(rank < 4 && "Unfortunately we do not support data with more than 3 dimensions. This is where you can help!!");
	
	// Initialize everything to zero.
    std::fill_n(data, num, 0);
	
	if (rank == 1){
		memcpy(data, img->data, sizeof(double)*img->num_data*img->num_channels*img->size_data.prod());
	}
	
	if (rank == 2){
		for (int i=0; i<img->num_data; i++) {
			for (int j=0; j<img->num_channels; j++) {
				for (int k=0; k<img->size_data(1); k++) {
					memcpy(data+(i*img->size_data_freq(1)*img->num_channels+j*img->size_data_freq(1)+k)*img->size_data_freq(0), img->data+(i*img->size_data(1)*img->num_channels+j*img->size_data(1)+k)*img->size_data(0), sizeof(double)*img->size_data(0));
				}
			}
		}
	}
	
	if (rank == 3){
		for (int i=0; i<img->num_data; i++) {
			for (int j=0; j<img->num_channels; j++) {
				for (int k=0; k<img->size_data(2); k++) {
					for (int l=0; l<img->size_data(1); l++) {
						memcpy(data+(i*img->size_data_freq(1)*img->size_data_freq(2)*img->num_channels+j*img->size_data_freq(1)*img->size_data_freq(2)+k*img->size_data_freq(1)+l)*img->size_data_freq(0), img->data+(i*img->size_data(1)*img->size_data(2)*img->num_channels+j*img->size_data(1)*img->size_data(2)+k*img->size_data(1)+l)*img->size_data(0),sizeof(double)*img->size_data(0));
					}
				}
			}
		}
	}
}

void CCorrelationFilters::fft_data(struct CDataStruct *img)
{	
	// Need to make this work with fftw threads, there seem to be some compiling and linking errors.
	
	int num_dim = img->size_data.size();
	int rank = num_dim;
	int *n = new int(num_dim);
	int *m = new int(num_dim);
	
	for (int i=0; i<num_dim; i++) {
		n[i] = img->size_data[i];
		m[i] = img->size_data_freq[i];
	}
	
	//int numCPU = 2; Is there is a way to automatically figure out the number of CPUs?
	//fftw_init_threads();
	//fftw_plan_with_nthreads(numCPU);
	
	fftw_plan plan;
	fftw_complex *out;
	
	int val = memcmp(n, m, num_dim*sizeof(int));
	img->num_elements_freq = (img->size_data_freq.prod()/img->size_data_freq(num_dim-1))*(img->size_data_freq(num_dim-1)/2+1);
    double scale_factor = sqrt(img->size_data_freq.prod());
    
    int istride = 1;
    int ostride = 1;
    int idist = img->size_data_freq.prod();
    int odist = (img->size_data_freq.prod()/img->size_data_freq(num_dim-1))*(img->size_data_freq(num_dim-1)/2+1);
	
	if (val != 0){
		// If the FFT size is NOT the same as the size of the data, zero pad the data.
		
		int num_channels = img->num_channels;
		double *data;
		data = new double[num_channels*img->size_data_freq.prod()];
		int howmany = num_channels;
		
		delete[] img->data_freq;
		img->data_freq = new complex<double>[odist*num_channels*img->num_data];
		
		out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*odist*num_channels);
		plan = fftw_plan_many_dft_r2c(rank, m, howmany, data, NULL, istride, idist, out, NULL, ostride, odist, FFTW_ESTIMATE);
				
		CDataStruct img1;
		img1.size_data = img->size_data;
		img1.size_data_freq = img->size_data_freq;
		img1.num_data = 1;
		img1.num_channels = img->num_channels;
		img1.data = new double[num_channels*img->size_data.prod()];
		
		for (int i=0; i<img->num_data; i++) {
			memcpy(img1.data,img->data+i*num_channels*img->size_data.prod(),sizeof(double)*num_channels*img->size_data.prod());
			zero_pad_data(data, &img1);
			fftw_execute(plan);
			memcpy(img->data_freq+i*odist*num_channels,reinterpret_cast<complex <double>*>(out),sizeof(complex<double>)*odist*num_channels);
		}
		
		for (int i=0; i<img->num_data; i++) {
			img->ptr_data_freq.push_back((img->data_freq+i*odist*num_channels));
		}
		
		fftw_destroy_plan(plan);
		fftw_free(out);
		delete[] data;
	}
	else{
		int howmany = img->num_data*img->num_channels;
		double *in = img->data;
		
		out = (fftw_complex*) fftwf_malloc(sizeof(fftw_complex)*odist*img->num_channels*img->num_data);
		plan = fftw_plan_many_dft_r2c(rank, n, howmany, in, NULL, istride, idist, out, NULL, ostride, odist, FFTW_ESTIMATE);
		fftw_execute(plan);
		img->data_freq = reinterpret_cast<complex <double>*>(out);
		
		for (int i=0; i<img->num_data; i++) {
			img->ptr_data_freq.push_back((img->data_freq+i*odist*img->num_channels));
		}
		fftw_destroy_plan(plan);
	}
    
    for (int i=0; i<img->num_data*img->num_channels*odist; i++){
        img->data_freq[i] = img->data_freq[i]/scale_factor;
    }
	
	//fftw_cleanup_threads();
	delete[] n;
	delete[] m;
}

void CCorrelationFilters::ifft_data(struct CDataStruct *img)
{
	// Need to make this work with fftw threads, there seem to be some compiling and linking errors.
	
	int num_dim = img->size_data.size();
	int rank = num_dim;
	int *n = new int(num_dim);
	int *m = new int(num_dim);
	
	for (int i=0; i<num_dim; i++) {
		n[i] = img->size_data(i);
		m[i] = img->size_data_freq(i);
	}
	
	fftw_plan plan;
	fftw_complex *in;
	double *out;
	
	int howmany = img->num_data*img->num_channels;
	in = reinterpret_cast<fftw_complex *>(img->data_freq);
	int istride = 1;
	int ostride = 1;
	int odist = img->size_data_freq.prod();
	int idist = (img->size_data_freq.prod()/img->size_data_freq(num_dim-1))*(img->size_data_freq(num_dim-1)/2+1);
	
	out = new double[odist*img->num_channels*img->num_data];
	plan = fftw_plan_many_dft_c2r(rank, m, howmany, in, NULL, istride, idist, out, NULL, ostride, odist, FFTW_ESTIMATE);
	
	fftw_execute(plan);
	img->data = out;
    
    double scale_factor = sqrt(odist);
    for (int i=0; i<odist*img->num_channels*img->num_data; i++) {
        img->data[i] = img->data[i]/scale_factor;
    }
	
	for (int i=0; i<img->num_data; i++) {
		img->ptr_data.push_back((img->data+i*odist*img->num_channels));
	}
	
	fftw_destroy_plan(plan);
	delete[] n;
	delete[] m;
}

void CCorrelationFilters::fusion_matrix_multiply(Eigen::ArrayXXcd &out, Eigen::ArrayXXcd &A, Eigen::ArrayXXcd &B, Eigen::Vector2i num_blocks_A, Eigen::Vector2i num_blocks_B)
{
	/* This function seems to be much slower (more than 10x) than the corresponding
	 MATLAB function. Needs heavy optimization. */
	
	int size_blocks = B.rows();
	int blocks_B = B.cols();
	int count;
	
	out = out*0;
	
	int index = 0;
	for (int i=0;i<num_blocks_A[0];i++){
		for (int j=0;j<num_blocks_B[1];j++){
			count = 0;
			for (int k=j;k<blocks_B;k=k+num_blocks_B[1]){
				out.block(0,index,size_blocks,1) += A.block(0,i*num_blocks_A[1]+count,size_blocks,1)*B.block(0,k,size_blocks,1);
				count++;
			}
			index++;
		}
	}
}

void CCorrelationFilters::fusion_matrix_inverse(Eigen::ArrayXXcd &X, Eigen::MatrixXi indices)
{
	/* We compute the inverse of the fusion matrix by using the schur complement
	 * on our data structure. We use an inplace implementation therefore it is quite
	 * memory efficient as well.
	 */
	
    int num_blocks=0;
    int total_blocks=0;
    int size_blocks = X.rows();
    
    if (indices.rows() != indices.cols()){
        std::cout << "Something Wrong" << std::endl;
        return;
    }
    else
    {
        num_blocks = indices.rows();
    }

    if (num_blocks == 1){
        X = X.inverse();
        return;
    }
	
	if (num_blocks == 2)
	{
        Eigen::ArrayXXcd temp1;
        Eigen::ArrayXXcd temp2;
        
		Eigen::ArrayXXcd DC(size_blocks,1);
		Eigen::ArrayXXcd BD(size_blocks,1);
		Eigen::ArrayXXcd BDC(size_blocks,1);
		Eigen::ArrayXXcd ABDC(size_blocks,1);
		
		DC = X.block(0,indices(1,0),size_blocks,1)/X.block(0,indices(1,1),size_blocks,1);
		BD = X.block(0,indices(0,1),size_blocks,1)/X.block(0,indices(1,1),size_blocks,1);
		BDC = X.block(0,indices(0,1),size_blocks,1)*DC;
		ABDC = (X.block(0,indices(0,0),size_blocks,1)-BDC).inverse();
		
		X.block(0,indices(0,0),size_blocks,1) = ABDC;
		X.block(0,indices(0,1),size_blocks,1) = -ABDC*BD;
		X.block(0,indices(1,0),size_blocks,1) = -DC*ABDC;
		X.block(0,indices(1,1),size_blocks,1) = 1/X.block(0,indices(1,1),size_blocks,1) - X.block(0,indices(1,0),size_blocks,1)*BD;
	}
	else
	{
        int num_B = num_blocks-1;
        int num_C = num_blocks-1;
        
        Eigen::MatrixXi ind_A;
        Eigen::MatrixXi ind_B;
        Eigen::MatrixXi ind_C;
        Eigen::MatrixXi ind_D;
        
        total_blocks = num_blocks*num_blocks;
        ind_D = indices.block(num_blocks-1,num_blocks-1,1,1);
        ind_B = indices.block(0,num_blocks-1,num_blocks-1,1);
        ind_C = indices.block(num_blocks-1,0,1,num_blocks-1);
        ind_A = indices.block(0,0,num_blocks-1,num_blocks-1).transpose();
        num_blocks = num_blocks-1;
		
		Eigen::ArrayXXcd D = Eigen::ArrayXXcd::Zero(size_blocks,1);
		Eigen::ArrayXXcd DC = Eigen::ArrayXXcd::Zero(size_blocks,num_blocks);
		Eigen::ArrayXXcd BD = Eigen::ArrayXXcd::Zero(size_blocks,num_blocks);
		Eigen::ArrayXXcd BDC = Eigen::ArrayXXcd::Zero(size_blocks,num_blocks*num_blocks);
		Eigen::ArrayXXcd ABDC = Eigen::ArrayXXcd::Zero(size_blocks,num_blocks*num_blocks);
		
		Eigen::ArrayXXcd tmp_B = Eigen::ArrayXXcd::Zero(size_blocks,num_B);
		Eigen::ArrayXXcd tmp_C = Eigen::ArrayXXcd::Zero(size_blocks,num_C);
		Eigen::ArrayXXcd tmp_D = Eigen::ArrayXXcd::Zero(size_blocks,1);
		
		for(int i=0;i<num_C;i++){
			tmp_C.block(0,i,size_blocks,1) = X.block(0,ind_C(0,i),size_blocks,1);
		}
		for(int i=0;i<num_B;i++){
			tmp_B.block(0,i,size_blocks,1) = X.block(0,ind_B(i,0),size_blocks,1);
		}
		
		D.block(0,0,size_blocks,1) = X.block(0,total_blocks-1,size_blocks,1).inverse();
		
		Eigen::Vector2i num_blocks_1, num_blocks_2;
		num_blocks_1 << 1,1;
		num_blocks_2 << 1,num_blocks;
		fusion_matrix_multiply(DC,D,tmp_C,num_blocks_1,num_blocks_2);
		
		num_blocks_1 << num_blocks,1;
		num_blocks_2 << 1,1;
		fusion_matrix_multiply(BD,tmp_B,D,num_blocks_1,num_blocks_2);
		
		num_blocks_1 << num_blocks,1;
		num_blocks_2 << 1,num_blocks;
		fusion_matrix_multiply(BDC,tmp_B,DC,num_blocks_1,num_blocks_2);
        
        for(int i=0;i<ind_A.rows();i++){
            for(int j=0;j<ind_A.cols();j++){
                X.block(0,ind_A(i,j),size_blocks,1) -= BDC.block(0,i,size_blocks,1);
            }
		}
		
		fusion_matrix_inverse(X,ind_A);
		
		tmp_B = Eigen::ArrayXXcd::Zero(size_blocks,num_B);
		tmp_C = Eigen::ArrayXXcd::Zero(size_blocks,num_C);
		tmp_D = Eigen::ArrayXXcd::Zero(size_blocks,1);
		
		num_blocks_1 << num_blocks,num_blocks;
		num_blocks_2 << num_blocks,1;
		fusion_matrix_multiply(tmp_B,ABDC,BD,num_blocks_1,num_blocks_2);
		for(int i=0;i<num_B;i++){
			X.block(0,ind_B(i,0),size_blocks,1) = -tmp_B.block(0,i,size_blocks,1);
		}
		
		num_blocks_1 << 1,num_blocks;
		num_blocks_2 << num_blocks,num_blocks;
		fusion_matrix_multiply(tmp_C,DC,ABDC,num_blocks_1,num_blocks_2);
		for(int i=0;i<num_C;i++){
			X.block(0,ind_C(0,i),size_blocks,1) = -tmp_C.block(0,i,size_blocks,1);
		}
		
		tmp_C = Eigen::ArrayXXcd::Zero(size_blocks,num_C);
		num_blocks_1 << 1,num_blocks;
		num_blocks_2 << num_blocks,num_blocks;
		fusion_matrix_multiply(tmp_C,DC,ABDC,num_blocks_1,num_blocks_2);
		
		num_blocks_1 << 1,num_blocks;
		num_blocks_2 << num_blocks,1;
		fusion_matrix_multiply(tmp_D,tmp_C,BD,num_blocks_1,num_blocks_2);
		tmp_D += D;
		X.block(0,total_blocks-1,size_blocks,1) = tmp_D.block(0,0,size_blocks,1);
	}
}

void CCorrelationFilters::compute_psd_matrix(struct CDataStruct *img, struct CParamStruct *params)
{
	int index;
	double temp_val;
    img->D = Eigen::ArrayXXcd::Zero(img->num_elements_freq,img->num_channels*img->num_channels);
	Eigen::Map<Eigen::ArrayXXcd> X(img->data_freq,img->num_elements_freq*img->num_channels,img->num_data);
	Eigen::ArrayXXcd temp = Eigen::ArrayXXcd::Zero(img->num_elements_freq,img->num_channels);
	Eigen::ArrayXXcd temp1 = Eigen::ArrayXXcd::Zero(img->num_elements_freq,1);
	Eigen::ArrayXXcd temp2 = Eigen::ArrayXXcd::Zero(img->num_elements_freq,1);
	
	// If not set default to 1
	if (params->wpos < 1) params->wpos = 1;
	
	switch ((int) params->whiten_flag)
	{
        // Case 0: Use only positive images to compute the pre-whitening matrix
		case 0:
			for (int k=0;k<img->num_data;k++){
				temp = X.block(0,k,img->num_elements_freq*img->num_channels,1);
				temp.resize(img->num_elements_freq,img->num_channels);
				if (img->labels[k] == 1)
				{
					index = 0;
					for (int i=0; i<img->num_channels; i++) {
						temp1 = temp.block(0,i,img->num_elements_freq,1);
						for (int j=0; j<img->num_channels; j++) {
							temp2 = temp.block(0,j,img->num_elements_freq,1);
							img->D.block(0,index,img->num_elements_freq,1) += params->wpos*(temp1.conjugate()*temp2);
							index++;
						}
					}
				}
			}
			
        // Case 1: Use only negative images to compute the pre-whitening matrix
		case 1:
			for (int k=0;k<img->num_data;k++){
				temp = X.block(0,k,img->num_elements_freq*img->num_channels,1);
				temp.resize(img->num_elements_freq,img->num_channels);
				if (img->labels[k] == -1)
				{
					index = 0;
					for (int i=0; i<img->num_channels; i++) {
						temp1 = temp.block(0,i,img->num_elements_freq,1);
						for (int j=0; j<img->num_channels; j++) {
							temp2 = temp.block(0,j,img->num_elements_freq,1);
							img->D.block(0,index,img->num_elements_freq,1) += (temp1.conjugate()*temp2);
							index++;
						}
					}
				}
			}
			
        // Case 2: Use all images to compute the pre-whitening matrix, usually gets you the best peformance
		case 2:
			for (int k=0;k<img->num_data;k++){
				temp = X.block(0,k,img->num_elements_freq*img->num_channels,1);
				temp.resize(img->num_elements_freq,img->num_channels);
				if (img->labels[k] == 1)
				{
					index = 0;
					for (int i=0; i<img->num_channels; i++) {
						temp1 = temp.block(0,i,img->num_elements_freq,1);
						for (int j=0; j<img->num_channels; j++) {
							temp2 = temp.block(0,j,img->num_elements_freq,1);
							img->D.block(0,index,img->num_elements_freq,1) += params->wpos*(temp1.conjugate()*temp2);
							index++;
						}
					}
				}
				else
				{
					index = 0;
					for (int i=0; i<img->num_channels; i++) {
						temp1 = temp.block(0,i,img->num_elements_freq,1);
						for (int j=0; j<img->num_channels; j++) {
							temp2 = temp.block(0,j,img->num_elements_freq,1);
							img->D.block(0,index,img->num_elements_freq,1) += (temp1.conjugate()*temp2);
							index++;
						}
					}
				}
			}
	}
	
	temp_val = img->D.real().maxCoeff();
	img->D.real() = (img->D.real())/temp_val;
	img->D.imag() = (img->D.imag())/temp_val;
    img->psd_flag = true;
}

void CCorrelationFilters::compute_inverse_psd_matrix(struct CDataStruct *img, struct CParamStruct *params)
{
    img->S = params->beta*img->D;
	for (int i=0; i<img->num_channels*img->num_channels; i=i+img->num_channels+1) {
		img->S.block(0,i,img->num_elements_freq,1) += Eigen::ArrayXXcd::Constant(img->num_elements_freq,1,params->alpha);
	}
	
    img->Sinv = img->S;
	Eigen::MatrixXi num_blocks = Eigen::MatrixXi::Zero(img->num_channels,img->num_channels);

    int index=0;
    for (int i=0; i<img->num_channels; i++){
        for (int j=0; j<img->num_channels; j++){
            num_blocks(i,j) = index;
            index++;
        }
    }
    
	fusion_matrix_inverse(img->Sinv, num_blocks);
    img->inv_psd_flag = true;
}

void CCorrelationFilters::build_uotsdf(struct CDataStruct *img, struct CParamStruct *params, struct CFilterStruct *filt)
{
	/*
	 * This function implements the correlation filter design proposed in the following publications.
     *
	 * [1] Vishnu Naresh Boddeti, Takeo Kanade and B.V.K. Vijaya Kumar, "Correlation Filters for Object Alignment," CVPR 2013.
     *
     * [2] A. Mahalanobis, B. V. K. Vijaya Kumar, S. Song, S. Sims, and J. Epperson. Unconstrained correlation filters. Applied Optics, 1994.
     
	 * Notes: This is currently the fastest Correlation Filter design to train, and is highly amenable for real-time and online learning or for object tracking. While in [1] we use this filter with HOG features, the filter design is general enough to be used with any other vector feature representation, for example a Gabor filter bank. Setting the filter parameter params->alpha=0 results in the unconstrained MACE filter.
	 */
	
	int num_pos=0,num_neg=0;
	
	filt->params = *params;
	filt->filter.size_data = params->size_filt_freq;
	filt->filter.size_data_freq = params->size_filt_freq;
	
	filt->filter.num_elements_freq = img->num_elements_freq;
	params->num_elements_freq = img->num_elements_freq;
	filt->filter.data_freq = new complex<double>[img->num_elements_freq*img->num_channels];
	
    Eigen::ArrayXXcd filt_freq = Eigen::ArrayXcd::Zero(params->num_elements_freq*img->num_channels,1);
	Eigen::ArrayXXcd pos_mean_freq = Eigen::ArrayXcd::Zero(params->num_elements_freq*img->num_channels,1);
    Eigen::ArrayXXcd neg_mean_freq = Eigen::ArrayXcd::Zero(params->num_elements_freq*img->num_channels,1);
	
	// If not set default to 1
	if (params->wpos < 1) params->wpos = 1;
	filt->params.wpos = params->wpos;
	
	Eigen::Map<Eigen::ArrayXXcd> X(img->data_freq,img->num_elements_freq*img->num_channels,img->num_data);
	
	Eigen::ArrayXXcd temp1 = Eigen::ArrayXXcd::Zero(img->num_elements_freq,img->num_channels);
	Eigen::ArrayXXcd temp2 = Eigen::ArrayXXcd::Zero(img->num_elements_freq,img->num_channels);
	Eigen::Vector2i num_blocks_1, num_blocks_2;
	
	num_blocks_1 << img->num_channels,img->num_channels;
	num_blocks_2 << img->num_channels,1;
	
	switch ((int) params->neg_flag)
	{
        // Case 0: Use only positive images, usually super fast.
		case 0:
			for (int k=0;k<img->num_data;k++){
				if (img->labels[k] == 1)
				{
					pos_mean_freq += X.block(0,k,img->num_elements_freq*img->num_channels,1)*params->wpos;
					num_pos++;
				}
			}
        // Case 1: Use only negative images, useful for online learning purposes.
		case 1:
			for (int k=0;k<img->num_data;k++){
				if (img->labels[k] == -1)
				{
					neg_mean_freq += X.block(0,k,img->num_elements_freq*img->num_channels,1);
					num_neg++;
				}
			}
        // Case 2: Use both positive and negative images, usually gets you the best performance.
		case 2:
			for (int k=0;k<img->num_data;k++){
				if (img->labels[k] == 1)
				{
					pos_mean_freq += X.block(0,k,img->num_elements_freq*img->num_channels,1)*params->wpos;
					num_pos++;
				}
				else
				{
					neg_mean_freq += X.block(0,k,img->num_elements_freq*img->num_channels,1);
					num_neg++;
				}
			}
	}
	
    filt_freq = pos_mean_freq/(num_pos*params->wpos)-neg_mean_freq/max(num_neg,1);
	filt_freq.resize(img->num_elements_freq,img->num_channels);
	fusion_matrix_multiply(temp1, img->Sinv, filt_freq, num_blocks_1, num_blocks_2);
	temp1.resize(img->num_elements_freq*img->num_channels,1);
	Eigen::Map<Eigen::ArrayXcd>(filt->filter.data_freq,img->num_elements_freq*img->num_channels) = temp1;
	
	filt->filter.num_data = 1;
	filt->filter.num_channels = img->num_channels;
	filt->filter.ptr_data.reserve(filt->filter.num_data);
	filt->filter.ptr_data_freq.reserve(filt->filter.num_data);
	ifft_data(&filt->filter);
    fft_data(&filt->filter);
}

void CCorrelationFilters::build_otsdf(struct CDataStruct *img, struct CParamStruct *params, struct CFilterStruct *filt)
{
    /*
	 * This function implements the correlation filter design proposed in the following publications.
	 * 
     * [1] Optimal trade-off synthetic discriminant function filters for arbitrary devices, B.V.K.Kumar, D.W.Carlson, A.Mahalanobis - Optics Letters, 1994.
	 *
	 * [2] Jason Thornton, "Matching deformed and occluded iris patterns: a probabilistic model based on discriminative cues." PhD thesis, Carnegie Mellon University, Pittsburgh, PA, USA, 2007.
	 *
	 * [3] Vishnu Naresh Boddeti, Jonathon M Smereka, and B. V. K. Vijaya Kumar, "A comparative evaluation of iris and ocular recognition methods on challenging ocular images." IJCB, 2011.
	 *
     * [4] A. Mahalanobis, B.V.K. Kumar, D. Casasent, "Minimum average correlation energy filters," Applied Optics, 1987
     *
	 * Notes: This filter design is good when the dimensionality of the data is greater than the training sample size. Setting the filter parameter params->alpha=0 results in the popular MACE filter. However, it is usually better to set alpha to a small number rather than setting it to 0. If you use MACE cite [4].
	 */

	filt->params = *params;
	filt->filter.size_data = params->size_filt_freq;
	filt->filter.size_data_freq = params->size_filt_freq;

	filt->filter.num_elements_freq = img->num_elements_freq;
	params->num_elements_freq = img->num_elements_freq;
	filt->filter.data_freq = new complex<double>[img->num_elements_freq*img->num_channels];
	
	Eigen::ArrayXcd filt_freq = Eigen::ArrayXcd::Zero(params->num_elements_freq*img->num_channels);
	
	// If not set default to 1
	if (params->wpos < 1) params->wpos = 1;
	filt->params.wpos = params->wpos;
	
	compute_psd_matrix(img, params);
	Eigen::MatrixXcd Y = Eigen::MatrixXcd::Zero(img->num_elements_freq*img->num_channels,img->num_data);
	Eigen::MatrixXcd u = Eigen::MatrixXcd::Zero(img->num_data,1);
	Eigen::MatrixXcd temp = Eigen::MatrixXcd::Zero(img->num_data,img->num_data);
	Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(img->num_data,img->num_data);
	
	Eigen::Map<Eigen::MatrixXcd> X(img->data_freq,img->num_elements_freq*img->num_channels,img->num_data);
	
	Eigen::ArrayXXcd temp1 = Eigen::ArrayXXcd::Zero(img->num_elements_freq,img->num_channels);
	Eigen::ArrayXXcd temp2 = Eigen::ArrayXXcd::Zero(img->num_elements_freq,img->num_channels);
	Eigen::Vector2i num_blocks_1, num_blocks_2;
	
	num_blocks_1 << img->num_channels,img->num_channels;
	num_blocks_2 << img->num_channels,1;
	
	for (int k=0;k<img->num_data;k++){
        temp2 = X.block(0,k,img->num_elements_freq*img->num_channels,1).array();
        temp2.resize(img->num_elements_freq,img->num_channels);
        fusion_matrix_multiply(temp1, img->Sinv, temp2, num_blocks_1, num_blocks_2);
        temp1.resize(img->num_elements_freq*img->num_channels,1);
        Y.block(0,k,img->num_elements_freq*img->num_channels,1) = temp1.matrix();
        temp1.resize(img->num_elements_freq,img->num_channels);
        
		if (img->labels[k] == 1)
		{
			u(k) = std::complex<double>(params->wpos,0);
		}
		else
		{
			u(k) = std::complex<double>(1,0);
		}
	}

	temp = X.conjugate().transpose()*Y;
	temp = temp.ldlt().solve(u);
	filt_freq = Y*temp;
	
	Y.resize(0,0);
	
	Eigen::Map<Eigen::ArrayXcd>(filt->filter.data_freq,img->num_elements_freq*img->num_channels) = filt_freq;
	filt->filter.num_data = 1;
	filt->filter.num_channels = img->num_channels;
	filt->filter.ptr_data.reserve(filt->filter.num_data);
	filt->filter.ptr_data_freq.reserve(filt->filter.num_data);
	ifft_data(&filt->filter);
    fft_data(&filt->filter);
}

void CCorrelationFilters::build_mmcf(struct CDataStruct *img, struct CParamStruct *params, struct CFilterStruct *filt)
{
    /*
	 * This function calls the correlation filter design proposed in the following publications.
     *
	 * A. Rodriguez, Vishnu Naresh Boddeti, B.V.K. Vijaya Kumar and A. Mahalanobis, "Maximum Margin Correlation Filter: A New Approach for Localization and Classification", IEEE Transactions on Image Processing, 2012.
     *
	 * Vishnu Naresh Boddeti, "Advances in Correlation Filters: Vector Features, Structured Prediction and Shape Alignment" PhD thesis, Carnegie Mellon University, Pittsburgh, PA, USA, 2012.
     *
	 * Vishnu Naresh Boddeti and B.V.K. Vijaya Kumar, "Maximum Margin Vector Correlation Filters," Arxiv 1404.6031 (April 2014).
	 *
	 * Notes: This currently the best performing Correlation Filter design, especially when the training sample size is larger than the dimensionality of the data.
	 */
	
	filt->params = *params;
	filt->filter.size_data = params->size_filt_freq;
	filt->filter.size_data_freq = params->size_filt_freq;
	
	filt->filter.num_elements_freq = img->num_elements_freq;
	params->num_elements_freq = img->num_elements_freq;
	filt->filter.data_freq = new complex<double>[img->num_elements_freq*img->num_channels];
	
	Eigen::ArrayXcd filt_freq = Eigen::ArrayXcd::Zero(params->num_elements_freq*img->num_channels);
	
	// If not set default to 1
	if (params->wpos < 1) params->wpos = 1;
	filt->params.wpos = params->wpos;
	
	compute_psd_matrix(img, params);
	Eigen::MatrixXcd Y = Eigen::MatrixXcd::Zero(img->num_elements_freq*img->num_channels,img->num_data);
	Eigen::MatrixXcd u = Eigen::MatrixXcd::Zero(img->num_data,1);
	Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(img->num_data,img->num_data);
	
	Eigen::Map<Eigen::MatrixXcd> X(img->data_freq,img->num_elements_freq*img->num_channels,img->num_data);
	
	Eigen::ArrayXXcd temp1 = Eigen::ArrayXXcd::Zero(img->num_elements_freq,img->num_channels);
	Eigen::ArrayXXcd temp2 = Eigen::ArrayXXcd::Zero(img->num_elements_freq,img->num_channels);
	Eigen::Vector2i num_blocks_1, num_blocks_2;
	
	num_blocks_1 << img->num_channels,img->num_channels;
	num_blocks_2 << img->num_channels,1;
	
	for (int k=0;k<img->num_data;k++){
        
        temp2 = X.block(0,k,img->num_elements_freq*img->num_channels,1).array();
        temp2.resize(img->num_elements_freq,img->num_channels);
        fusion_matrix_multiply(temp1, img->Sinv, temp2, num_blocks_1, num_blocks_2);
        temp1.resize(img->num_elements_freq*img->num_channels,1);
        Y.block(0,k,img->num_elements_freq*img->num_channels,1) = temp1.matrix();
        temp1.resize(img->num_elements_freq,img->num_channels);
        
		if (img->labels[k] == 1)
		{
			u(k) = std::complex<double>(params->wpos,0);
		}
		else
		{
			u(k) = std::complex<double>(-1,0);
		}
	}
	
	esvm::SVMClassifier libsvm;
	
	libsvm.setC(params->C);
	libsvm.setKernel(params->kernel_type);
	libsvm.setWpos(params->wpos);
	
	temp = (X.conjugate().transpose()*Y).real();
	Eigen::Map<Eigen::MatrixXd> y(img->labels,img->num_data,1);
	
	libsvm.train(temp, y);
	temp.resize(0,0);
	
	int nSV;
	libsvm.getNSV(&nSV);
	Eigen::VectorXi sv_indices = Eigen::VectorXi::Zero(nSV);
	Eigen::VectorXd sv_coef = Eigen::VectorXd::Zero(nSV);
	libsvm.getSI(sv_indices);
	libsvm.getCoeff(sv_coef);
	
	for (int k=0; k<nSV; k++) {
		filt_freq += (Y.block(0,sv_indices[k]-1,img->num_elements_freq*img->num_channels,1)*sv_coef[k]).array();
	}
	
	Y.resize(0,0);
	
	Eigen::Map<Eigen::ArrayXcd>(filt->filter.data_freq,img->num_elements_freq*img->num_channels) = filt_freq;
	filt->filter.num_data = 1;
	filt->filter.num_channels = img->num_channels;
	filt->filter.ptr_data.reserve(filt->filter.num_data);
	filt->filter.ptr_data_freq.reserve(filt->filter.num_data);
	ifft_data(&filt->filter);
    fft_data(&filt->filter);
}

void CCorrelationFilters::apply_filter(struct CDataStruct *corr, struct CDataStruct *img, struct CFilterStruct *filt)
{
	int num_dim1 = img->size_data.size();
	int num_dim2 = filt->filter.size_data.size();
	
	assert(num_dim1 == num_dim2 && "Something wrong, filter and image have different rank!!");
    assert(img->num_channels==filt->filter.num_channels && "Something wrong, filter and image have different number of channels!!");
	
	int *siz = new int(num_dim1);
	
	for (int i=0; i<num_dim1; i++) {
//		siz[i] = img->size_data(i)+filt->filter.size_data(i)-1;
        siz[i] = img->size_data(i);
	}
	
	Eigen::Map<Eigen::VectorXi> size_data_freq(siz,num_dim1);
	img->size_data_freq = size_data_freq;
	filt->filter.size_data_freq = size_data_freq;
	
	fft_data(img);
	fft_data(&filt->filter);
    
	double *data;
	data = new double[img->num_data*filt->filter.num_data*img->size_data_freq.prod()];
	corr->data = data;
    corr->size_data_freq = img->size_data_freq;
    corr->size_data = corr->size_data_freq;
	
	int num_elements_freq = img->num_elements_freq;
	int num_elements_data = img->size_data_freq.prod();
	
	Eigen::ArrayXXcd tmp1 = Eigen::ArrayXcd::Zero(num_elements_freq*img->num_channels,1);
	Eigen::ArrayXXcd tmp2 = Eigen::ArrayXcd::Zero(num_elements_freq*filt->filter.num_channels,1);
	Eigen::ArrayXcd temp = Eigen::ArrayXcd::Zero(num_elements_freq);
	Eigen::Map<Eigen::ArrayXXcd> temp1(img->data_freq,num_elements_freq*img->num_channels,img->num_data);
	Eigen::Map<Eigen::ArrayXXcd> temp2(filt->filter.data_freq,num_elements_freq*filt->filter.num_channels,filt->filter.num_data);
	
	CDataStruct tmp;
	tmp.size_data = img->size_data;
	tmp.size_data_freq = img->size_data_freq;
	tmp.num_data = 1;
	tmp.num_channels = 1;
	tmp.ptr_data.reserve(tmp.num_data);
	tmp.ptr_data_freq.reserve(tmp.num_data);
	tmp.data_freq = new complex<double>[num_elements_freq];
	
	for (int i=0; i<img->num_data; i++) {
		tmp1 = temp1.block(0,i,num_elements_freq*img->num_channels,1);
		tmp1.resize(num_elements_freq,img->num_channels);
		for (int j=0; j<filt->filter.num_data; j++) {
			tmp2 = temp2.block(0,j,num_elements_freq*img->num_channels,1);
			tmp2.resize(num_elements_freq,img->num_channels);
            
			temp = temp*0;
			for (int k=0; k<img->num_channels; k++) {
				temp += tmp1.block(0,k,num_elements_freq,1)*tmp2.block(0,k,num_elements_freq,1).conjugate();
			}

			Eigen::Map<Eigen::ArrayXcd> (tmp.data_freq,num_elements_freq) = temp;
			ifft_data(&tmp);
			memcpy((corr->data + (i*filt->filter.num_data+j)*num_elements_data), tmp.data, sizeof(double)*num_elements_data);
			delete[] tmp.data;
		}
	}
	
	filt->filter.size_data_freq = filt->filter.size_data;
	img->size_data_freq = img->size_data;
}

void CCorrelationFilters::save_filter(struct CFilterStruct *filt, const char *filename)
{
	int rank = filt->filter.size_data.size();
	ofstream file(filename, ios::out|ios::binary);
	file.write((char*)&filt->filter.num_data, sizeof(int));
	file.write((char*)&filt->filter.num_channels, sizeof(int));
	file.write((char*)&filt->filter.num_elements_freq, sizeof(int));
	
	file.write((char*)&filt->params.C, sizeof(double));
	file.write((char*)&filt->params.alpha, sizeof(double));
	file.write((char*)&filt->params.beta, sizeof(double));
	file.write((char*)&filt->params.kernel_type, sizeof(int));
	file.write((char*)&filt->params.whiten_flag, sizeof(double));
	file.write((char*)&filt->params.neg_flag, sizeof(double));
	file.write((char*)&filt->params.wpos, sizeof(double));
    
	file.write((char*)&rank, sizeof(int));
	file.write((char*)filt->filter.size_data.data(), rank*sizeof(int));
	file.write((char*)filt->filter.size_data_freq.data(), rank*sizeof(int));
	file.write((char*)filt->filter.data, sizeof(double)*filt->filter.num_channels*filt->filter.size_data.prod()*filt->filter.num_data);
	file.write((char*)filt->filter.data_freq, sizeof(complex<double>)*filt->filter.num_channels*filt->filter.num_elements_freq*filt->filter.num_data);
	file.close();
}

void CCorrelationFilters::load_filter(struct CFilterStruct *filt, const char *filename)
{
	int rank;
	ifstream file(filename, ios::in|ios::binary);
	file.read((char*)&filt->filter.num_data, sizeof(int));
	file.read((char*)&filt->filter.num_channels, sizeof(int));
	file.read((char*)&filt->filter.num_elements_freq, sizeof(int));
	
	file.read((char*)&filt->params.C, sizeof(double));
	file.read((char*)&filt->params.alpha, sizeof(double));
	file.read((char*)&filt->params.beta, sizeof(double));
	file.read((char*)&filt->params.kernel_type, sizeof(int));
	file.read((char*)&filt->params.whiten_flag, sizeof(double));
	file.read((char*)&filt->params.neg_flag, sizeof(double));
	file.read((char*)&filt->params.wpos, sizeof(double));
	
	file.read((char*)&rank, sizeof(int));
	Eigen::VectorXi size_data = Eigen::VectorXi::Zero(rank);
	Eigen::VectorXi size_data_freq = Eigen::VectorXi::Zero(rank);
	
	file.read((char*)size_data.data(), rank*sizeof(int));
	file.read((char*)size_data_freq.data(), rank*sizeof(int));
	
	filt->filter.size_data = size_data;
	filt->filter.size_data_freq = size_data_freq;
	
	filt->filter.data = new double[filt->filter.num_channels*filt->filter.size_data.prod()*filt->filter.num_data];
	filt->filter.data_freq = new complex<double>[filt->filter.num_channels*filt->filter.num_elements_freq*filt->filter.num_data];
	
	file.read((char*)filt->filter.data, sizeof(double)*filt->filter.num_channels*filt->filter.size_data.prod()*filt->filter.num_data);
	file.read((char*)filt->filter.data_freq, sizeof(complex<double>)*filt->filter.num_channels*filt->filter.num_elements_freq*filt->filter.num_data);
	file.close();
	
	filt->filter.ptr_data.reserve(filt->filter.num_data);
	filt->filter.ptr_data_freq.reserve(filt->filter.num_data);
	
	for (int i=0; i<filt->filter.num_data; i++) {
		filt->filter.ptr_data[i] = (filt->filter.data + i*sizeof(double)*filt->filter.size_data.prod()*filt->filter.num_channels);
		filt->filter.ptr_data_freq[i] = (filt->filter.data_freq + i*sizeof(complex<double>)*filt->filter.num_elements_freq*filt->filter.num_channels);
	}
}