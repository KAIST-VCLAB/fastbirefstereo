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

#include "rectifier.h"

#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
	cv::UMat bO2D, bE2D, tformInd, invInd;
	std::vector<cv::UMat> handle(2);

	// Read LuT given by Birefractive stereo's model (http://vclab.kaist.ac.kr/siggraphasia2016p1/)
	handle[0] = cv::imread("resources/b_o2d_1.exr", cv::IMREAD_UNCHANGED).getUMat(cv::ACCESS_READ);
	handle[1] = cv::imread("resources/b_o2d_2.exr", cv::IMREAD_UNCHANGED).getUMat(cv::ACCESS_READ);
	cv::merge(handle, bO2D);

	handle[0] = cv::imread("resources/b_e2d_1.exr", cv::IMREAD_UNCHANGED).getUMat(cv::ACCESS_READ);
	handle[1] = cv::imread("resources/b_e2d_2.exr", cv::IMREAD_UNCHANGED).getUMat(cv::ACCESS_READ);
	cv::merge(handle, bE2D);

	// Rectification mapping via dynamic programming
	std::cout << "Disparity coeficient: f * baseline = " <<
		Rectifier::buildRectification(bO2D, bE2D, tformInd, invInd) << std::endl;
	std::cout << "Reverse rectification..." << std::endl;
	invInd = cv::UMat::zeros(bO2D.size(), CV_32FC2);
	Rectifier::reverseRectification(tformInd, invInd);

	// Write the rectification tables
	cv::split(tformInd, handle);
	cv::imwrite("resources/tform_ind_new1.exr", handle[0]);
	cv::imwrite("resources/tform_ind_new2.exr", handle[1]);
	cv::split(invInd, handle);
	cv::imwrite("resources/inv_ind_new1.exr", handle[0]);
	cv::imwrite("resources/inv_ind_new2.exr", handle[1]);

	return 0;
}
