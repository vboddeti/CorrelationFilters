/*
 *  main.cpp
 *  CorrelationFilters
 *
 *  Created by Vishnu Boddeti on 5/22/13.
 *	naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
 *  Copyright 2013 Carnegie Mellon University. All rights reserved.
 *
 */

#include "CCorrelationFilters.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

void display_data(double *data, int m, int n)
{
    int index = 0;
    for (int i=0; i<m; i++) {
        for (int j=0; j<m; j++) {
            std::cout << data[index] << " ";
            index++;
        }
        std::cout << "\n";
    }
}

void display_data_freq(complex<double> *data, int m, int n)
{
    int index = 0;
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            std::cout << data[index] << " ";
            index++;
        }
        std::cout << "\n";
    }
}

int main()
{
    int num_class = 1;
    int num_img_per_class = 1;
    int num_dim = 2;
    int num_channels = 1;
    int size_img[] = {112, 92};
    
    double *im, *labels;
    im = new double[num_class*num_img_per_class*num_channels*size_img[0]*size_img[1]];
    labels = new double[num_class*num_img_per_class]();
    std::fill_n(labels, num_class*num_img_per_class, -1.0);
    
    char filename[] = "/Users/vboddeti/Dropbox/CorrelationFiltersLocal/data/ORL/%d_%d.jpg";
    char name[2000];
    
    cv::Mat img, dimg, small_img;
    
    int index=0;
    for(int i=0; i<num_class; i++){
        for (int j=0; j<num_img_per_class; j++){
            std::sprintf(name, filename, i+1, j+1);
            img = cv::imread(name);
            img.convertTo( dimg, CV_64FC3, 1.0/255.0);
            for (int k=0; k<dimg.rows; k++){
                for (int l=0; l<dimg.cols; l++){
                    im[index] = dimg.at<double>(k,l);
                    index++;
                }
            }
        }
    }
    
//    int index;
//    int num_dim = 2;
//    int num_channels = 1;
//    int num_class = 1;
//    int num_img_per_class = 1;
//    
//    std::ifstream file;
//    std::string filename = "/Users/vboddeti/temp.bin";
//    file.open(filename, std::ios::in|std::ios::binary);
//    int m, n;
//    file.read((char *)&m, sizeof(int));
//    file.read((char *)&n, sizeof(int));
//    double *im;
//    im = new double[m*n];
//    file.read((char *)im, sizeof(double)*m*n);
//    file.close();
//    double *labels;
//    labels = new double[1];
//    int size_img[2];
//    size_img[0] = m;
//    size_img[1] = n;
    
    CDataStruct data;
    CCorrelationFilters CF;
    CF.initialize_data(&data, im, labels, num_class*num_img_per_class, size_img, num_dim, num_channels);
    
    CParamStruct filt_params;
    filt_params.C = 1;
    filt_params.alpha = 1e-3;
    filt_params.beta = 1-filt_params.alpha;
    filt_params.kernel_type = 0;
    filt_params.whiten_flag = 2;
    filt_params.neg_flag = 0;
    filt_params.wpos = 1;
    filt_params.size_filt = Eigen::VectorXi::Zero(0);
    filt_params.size_filt_freq = Eigen::VectorXi::Zero(0);
    filt_params.num_elements_freq = 0;
    filt_params.size_filt = data.size_data;
    filt_params.size_filt_freq = filt_params.size_filt;
    
    CF.prepare_data(&data, &filt_params);

    std::vector<CFilterStruct> mmcf;
    std::vector<CFilterStruct> uotsdf;
    
    mmcf.reserve(num_class);
    uotsdf.reserve(num_class);
    
    CDataStruct corrplane;
    CDataStruct temp;
    
    cv::Mat corr_output;
    double max_val;
    double min_val;
    
    std::ofstream file;
    file.open("/Users/vboddeti/corrplane.bin");
    
    for (int i=0; i<num_class; i++) {
        CFilterStruct temp1;
        CFilterStruct temp2;
        std::fill_n(data.labels, data.num_data, -1.0);
        std::fill_n(data.labels+i*num_img_per_class, num_img_per_class, 1.0);
        CF.build_uotsdf(&data, &filt_params, &temp1);
        uotsdf.push_back(temp1);

//        CF.build_mmcf(&data, &filt_params, &temp2);
//        mmcf.push_back(temp2);
        
        corr_output = cv::Mat(temp1.filter.size_data[0],temp1.filter.size_data[1],CV_64F,temp1.filter.data);
        
        
        for (int j=0; j<num_img_per_class; j++){
            index = i*num_img_per_class+j;
            CF.initialize_data(&temp, data.ptr_data[index], &data.labels[index], 1, size_img, 2, 1);
//            CF.normalize_data(&temp);
            CF.apply_filter(&corrplane, &temp, &uotsdf[i]);
            
            file.write((char*)&corrplane.size_data[0], sizeof(int));
            file.write((char*)&corrplane.size_data[1], sizeof(int));
            file.write((char*)corrplane.data, sizeof(double)*corrplane.size_data(0)*corrplane.size_data(1));
            
//            corr_output = cv::Mat(corrplane.size_data[0],corrplane.size_data[1],CV_64F,corrplane.data);
//            cv::minMaxLoc(corr_output, &min_val, &max_val);
//            corr_output = (corr_output-min_val)/(max_val-min_val);
//            cv::imshow("corrplane",corr_output);
//            cv::waitKey();
        }
    }
    
    file.close();

    std::cout << " " << std::endl;
	return 0;
}