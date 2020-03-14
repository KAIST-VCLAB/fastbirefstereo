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

#include <opencv2/imgproc.hpp>

// squared l2 norm between the cv::Point2f a and [bx, by]
#define POINTS_DIFF(a,bx,by) ((a.x-bx)*(a.x-bx)+(a.y-by)*(a.y-by))

float Rectifier::buildRectification(cv::UMat const & bO2D, cv::UMat const & bE2D,
	cv::UMat & tformInd, cv::UMat & invInd)
{
	// Add thirsty rows to prevent the image from being cropped
	tformInd = cv::UMat::zeros(cv::Size(bO2D.cols + 30, bO2D.rows), CV_32FC2);

	// Remove depth dependency
	cv::UMat bO2E;
	cv::subtract(bO2D, bE2D, bO2E);

	// Get what will be the horizontal baseline and normalise the disparities
	float baseline(cv::mean(bO2E)[0]);
	cv::divide(bO2E, baseline, bO2E);

	// Initialize the first column to identity mapping
	cv::Mat identityGrid(cv::Mat::zeros(cv::Size(1, tformInd.rows), CV_32FC2));
	for (int i(1); i < tformInd.rows; i++)
	{
		identityGrid.at<cv::Point2f>(i, 0) = cv::Point2f(0.f, float(i) * float(bO2E.rows) / float(tformInd.rows));
	}
	identityGrid.getUMat(cv::ACCESS_READ).copyTo(tformInd.col(0));

	// Dynamic programming rectification
	for (int j(1); j < tformInd.cols; j++)
	{
		// Get the current column local disparity to be mapped to horizontal
		cv::remap(bO2E, tformInd.col(j), tformInd.col(j - 1), cv::noArray(), cv::INTER_LINEAR);

		// Add to previous column (j - 1)
		cv::add(tformInd.col(j - 1), tformInd.col(j), tformInd.col(j));
	}

	return baseline;
}

void Rectifier::reverseRectification(cv::UMat const & tformInd, cv::UMat & invInd, double scale)
{
	cv::UMat tformIndGreater;
	cv::resize(tformInd, tformIndGreater, cv::Size(), scale, scale);
	cv::multiply(tformIndGreater, scale, tformIndGreater);
	cv::Mat tformIndMat(tformIndGreater.getMat(cv::ACCESS_READ)),
		invIndMat(cv::Mat::zeros(cv::Size(invInd.cols * scale, invInd.rows * scale), CV_32FC2)),
		bestDiff(cv::Mat::ones(tformIndMat.size(), CV_32FC1));

	for (int i(1); i < tformIndMat.rows - 1; i++)
	{
		for (int j(1); j < tformIndMat.cols - 1; j++)
		{
			// Get where the rectification map is pointing
			cv::Point2f currPos = tformIndMat.at<cv::Point2f>(i, j);
			cv::Point2i currIndices = cv::Point2i(currPos);

			// For our application, the boundary would not be used, 
			// it is therefore acceptable to not specify padding
			if (currIndices.x > 1 && currIndices.y > 1 && 
				currIndices.x < invIndMat.cols - 1 && currIndices.y < invIndMat.rows - 1)
			{
				// Check if the current pixel is the best match for the pointed one
				// and check this for the neighbours of the pointed one as well
				// This fills-in unmatched holes if their are any
				for (int k(0); k < 9; k++)
				{
					// neighbours in the reverse map
					int x(currIndices.x - 1 + k % 3), y(currIndices.y - 1 + k / 3);
					float diff = POINTS_DIFF(currPos, float(x), float(y)) / 2;

					// if d(T([i, j]), [x,y]) is lower than for other [x,y] and [i, j] couples,
					// set T^{-1}([x,y]) = [i, j]
					if (diff < bestDiff.at<float>(currIndices))
					{
						bestDiff.at<float>(y, x) = diff;
						invIndMat.at<cv::Point2f>(y, x) = cv::Point2f(float(j), float(i));
					}
				}
			}
		}
	}

	// Restore the original size 
	cv::resize(invIndMat, invIndMat, invInd.size());
	invInd = invIndMat.getUMat(cv::ACCESS_READ);
	cv::multiply(invInd, 1. / scale, invInd);
	cv::subtract(invInd, cv::Scalar(1.f, 1.f), invInd);
}
