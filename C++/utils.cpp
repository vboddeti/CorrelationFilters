/*
 *  utils.cpp
 *  CorrelationFilters
 *
 *  Created by Vishnu Boddeti on 7/1/13.
 *	naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
 *  Copyright 2013 Carnegie Mellon University. All rights reserved.
 *
 */

#include "utils.h"

#define cvtype CV_64F

void myunion(std::vector<int> &v, std::vector<int> &A, std::vector<int> &B)
{
	std::vector<int>::iterator it;
	sort(A.begin(),A.end());
	sort(B.begin(),B.end());
	it = set_union(A.begin(),A.end(),B.begin(),B.end(),v.begin());
	v.resize(it-v.begin());
}

void mysetdiff(std::vector<int> &v, std::vector<int> &A, std::vector<int> &B)
{
	std::vector<int>::iterator it;
	sort(A.begin(),A.end());
	sort(B.begin(),B.end());
	it = set_difference(A.begin(),A.end(),B.begin(),B.end(),v.begin());
	v.resize(it-v.begin());
}

// type conversion from darwin framework edited to work properly (opencv is row-major, while eigen is column-major)
Eigen::MatrixXd cvMat2eigen(const CvMat *m) {
	// TODO: make this work for any scalar type
	Eigen::MatrixXd d;
	
	if(m == NULL){ return d; }
	
	d.resize(m->rows, m->cols);
	
	switch (cvGetElemType(m)) {
		case CV_8UC1:
		{
			int ct = 0;
			const unsigned char *p = (unsigned char *)CV_MAT_ELEM_PTR(*m, 0, 0);
			for(int i = 0; i < m->rows; i++) {
				for(int j = 0; j < m->cols; j++) {
					d(i,j) = (double)p[ct]; ct++;
				}
			}
		}
			break;
			
		case CV_8SC1:
		{
			int ct = 0;
			const char *p = (char *)CV_MAT_ELEM_PTR(*m, 0, 0);
			for(int i = 0; i < m->rows; i++) {
				for(int j = 0; j < m->cols; j++) {
					d(i,j) = (double)p[ct]; ct++;
				}
			}
		}
			break;
			
		case CV_32SC1:
		{
			Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> temp;
			temp = Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >((int *)m->data.ptr, m->rows, m->cols);
			d = temp.cast<double>();
		}
			break;
			
		case CV_32FC1:
		{
			Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> temp;
			temp = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >((float *)m->data.ptr, m->rows, m->cols);
			d = temp.cast<double>();
		}
			break;
			
		case CV_64FC1:
		{
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> temp;
			temp = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >((double *)m->data.ptr, m->rows, m->cols);
			d = temp.cast<double>();
		}
			break;
			
		default:
			std::cout << "unrecognized openCV matrix type: " << cvGetElemType(m) << "\n";
			break;
	}
	
	return d;
}

// type conversion from darwin framework
CvMat *eigen2cvMat(const Eigen::MatrixXd &m, int mType) {
	// TODO: make this work for any scalar type
	CvMat *d = cvCreateMat(m.rows(), m.cols(), mType);
	if(d == NULL) return NULL;
	
	switch (mType) {
		case CV_8UC1:
		case CV_8SC1:
			std::cout << "not implemented yet\n";
			break;
			
		case CV_32SC1:
		{
			Eigen::MatrixXi temp = m.cast<int>();
			Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >((int *)d->data.ptr, d->rows, d->cols) = temp;
		}
			break;
			
		case CV_32FC1:
		{
			Eigen::MatrixXf temp = m.cast<float>();
			Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >((float *)d->data.ptr, d->rows, d->cols) = temp;
		}
			break;
			
		case CV_64FC1:
		{
			Eigen::MatrixXd temp = m.cast<double>();
			Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >((double *)d->data.ptr, d->rows, d->cols) = temp;
		}
			break;
			
		default:
			std::cout << "unrecognized openCV matrix type: " << mType << "\n";
			break;
	}
	
	return d;
}

void EigShowImg(const Eigen::MatrixXd &mat) {
	// TODO: make this work for any scalar type
	CvMat *convert = eigen2cvMat(mat, cvtype);
	cv::Mat image = convert;
	image.convertTo(image, CV_8U);
	cv::namedWindow("Display window", CV_WINDOW_AUTOSIZE); // Create a window for display.
	cv::imshow("Display window", image);                   // Show our image inside it.
	cv::waitKey(0);
}

void loadimages(std::vector<Eigen::MatrixXd> &imgs, std::string directory, int width, int height) {
	std::vector<std::string> Compat; // checks that the loaded images meet extensions specs
	/* Image file formats supported by OpenCV: BMP, DIB, JPEG, JPG, JPE, JP2, PNG, PBM, PGM, PPM, SR, RAS, TIFF, TIF */
	Compat.clear(); // backward to make it a bit faster to check
	Compat.push_back("pmb");  Compat.push_back("bid");  Compat.push_back("gepj");
	Compat.push_back("gpj");  Compat.push_back("epj");  Compat.push_back("gnp");
	Compat.push_back("mbp");  Compat.push_back("mgp");  Compat.push_back("mpp");
	Compat.push_back("rs");   Compat.push_back("sar");  Compat.push_back("ffit");
	Compat.push_back("fit");  Compat.push_back("2pj");
	DIR *d; struct dirent *dir;   // file pointer within directory
	d = opendir(directory.c_str());
	if(!d) {
		return;
	}
	
	std::vector<int> authimg;
	std::vector<int> impimg;
	
	int i, k, ct=0, ct2=0;
	std::string imgChk, name;
	std::string loadname;
	cv::Mat tmpimg, resizedimg;
	while((dir = readdir(d)) != NULL) { // not empty
		k = dir->d_reclen;
		if(k > 2 && dir->d_type == DT_REG) { // not a folder or '.', '..'
			imgChk.clear(); name.clear();
			name.assign(dir->d_name);
			for(i=name.size()-1; i>0; i--) { // get file extention (backward)
				if(!name.compare(i,1,".")) {
					break;
				} else {
					imgChk.push_back(tolower(name[i]));
				}
			}
			for(i=0; i<(int)(Compat.size()); i++) { // compare with compatibility list
				if(imgChk.compare(Compat[i]) == 0) {
					break;
				}
			}
			if(i < (int)(Compat.size())) { // image is of compatible format
				ct++;
				loadname = directory + name;
				std::cout << "Image " << ct << ":";
				tmpimg = cv::imread(loadname.c_str(), 0); // force to be grayscale
				if(!tmpimg.data){
					std::cout << "image data did not load properly for " << loadname << ", ";
				} else {
					Eigen::MatrixXd newmat; CvMat convert;
					std::cout << " Loaded " << name;
					
					if(!name.compare(0,4,"auth")) {
						// authentic image
						authimg.push_back(ct2);
					} else if(!name.compare(0,3,"imp")) {
						impimg.push_back(ct2);
					}
					
					if(tmpimg.rows != height || tmpimg.cols != width) {
						resize(tmpimg, resizedimg, cv::Size(width,height), 0, 0, cv::INTER_CUBIC);
						std::cout << ", Resized (" << tmpimg.rows << "x" << tmpimg.cols << "->" << height << "x" << width << ")";
						resizedimg.convertTo(resizedimg, cvtype);
						convert = resizedimg;
					} else {
						std::cout << ", No Resizing";
						tmpimg.convertTo(tmpimg, cvtype);
						convert = tmpimg;
					}
					newmat = cvMat2eigen(&convert);
					imgs.push_back(newmat);
					std::cout << ", Complete\n"; ct2++;
				}
			} else {
				std::cout << "Not Loading: " << name << "\n";
			}
		}
	}
	std::cout << "Loaded a total of " << ct << " images\n";
	closedir(d);
}