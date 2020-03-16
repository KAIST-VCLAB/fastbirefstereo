/****************************************************************************
- Codename: Single-shot Monocular RGB-D Imaging using Uneven Double Refraction (CVPR 2020)
- author: Andreas Meuleman (ameuleman@vclab.kaist.ac.kr)
- Institute: KAIST Visual Computing Laboratory
@InProceedings{Meuleman_2020_CVPR,
	author = {Andreas Meuleman and Seung-Hwan Baek and Felix Heide and Min H. Kim},
	title = {Single-shot Monocular RGB-D Imaging using Uneven Double Refraction},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {June},
	year = {2020}
}

Copyright (c) 2020 Andreas Meuleman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*****************************************************************************/

#include "depth_estimator.h"

#include <opencv2/opencv.hpp>

#define MIN_DEPTH 450.f
#define MAX_DEPTH 800.f
#define BASELINE -8013.f
#define TAU 0.286f

int main(int argc, char **argv)
{
	// Read the rectification tables
	cv::UMat tformInd, invInd;
	std::vector<cv::UMat> handle(2);

	handle[0] = cv::imread("resources/tform_ind1.exr", cv::IMREAD_UNCHANGED).getUMat(cv::ACCESS_READ);
	handle[1] = cv::imread("resources/tform_ind2.exr", cv::IMREAD_UNCHANGED).getUMat(cv::ACCESS_READ);
	cv::merge(handle, tformInd);
	
	handle[0] = cv::imread("resources/inv_ind1.exr", cv::IMREAD_UNCHANGED).getUMat(cv::ACCESS_READ);
	handle[1] = cv::imread("resources/inv_ind2.exr", cv::IMREAD_UNCHANGED).getUMat(cv::ACCESS_READ);
	cv::merge(handle, invInd);

	// Initialise and set parameters
	DepthEstimator depthEstimator(tformInd, invInd, MIN_DEPTH, MAX_DEPTH, BASELINE, TAU);
	tformInd.release();
	invInd.release();

	// Read the input image and run the algorithm
	cv::UMat in(cv::imread("resources/demo.png").getUMat(cv::ACCESS_READ));
	depthEstimator.setFrame(in.clone());

	cv::imshow("Restored", depthEstimator.getReconsImg());
	cv::imshow("Input", in);
	cv::imshow("Disparity map", depthEstimator.getDisparityMap());
	cv::waitKey(0);

	return 0;
}
