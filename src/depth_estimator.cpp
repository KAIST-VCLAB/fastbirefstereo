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

#include <iostream>
#include <fstream>

DepthEstimator::DepthEstimator(cv::UMat const & tformInd, cv::UMat const & invInd, 
	float minZ, float maxZ, float disparityCoef, float tau, float upsampling,
	double scaleMask, int winSize, unsigned char threshGrad, unsigned char threshCost):
	m_tau(tau),
	m_winSize(int(upsampling * winSize) + 1 - (int(upsampling * winSize) % 2)),
	m_threshGrad(threshGrad),
	m_threshCost(threshCost),
	m_disparityCoef(upsampling * disparityCoef),
	m_kernelGrad1((cv::Mat_<float>(3, 3) << 
		-6, 0, 6,
		-20, 0, 20,
		-6, 0, 6)),
	m_kernelGrad2((cv::Mat_<float>(3, 3) << 
		6, 0, -6,
		20, 0, -20,
		6, 0, -6))
{
	// Resize and optimise LuTs
	cv::UMat invIndMask, invIndHandle, tformIndHandle;
	cv::multiply(invInd, upsampling, invIndHandle);
	cv::resize(invIndHandle, invIndMask, cv::Size(), scaleMask, scaleMask);
	cv::resize(tformInd, tformIndHandle, cv::Size(), double(upsampling), double(upsampling));

	cv::convertMaps(invIndHandle, cv::noArray(), m_invInd1, m_invInd2, CV_16SC2);
	cv::convertMaps(tformIndHandle, cv::noArray(), m_tformInd1, m_tformInd2, CV_16SC2);
	cv::convertMaps(invIndMask, cv::noArray(), m_invIndMask1, m_invIndMask2, CV_16SC2);

	// Create the depth candidates 
	float a(m_disparityCoef / maxZ), b(m_disparityCoef / minZ);
	m_zCount = a - b + 1.5;
	float step((b - a) / (m_zCount - 1));
	m_disparities.resize(m_zCount);
	for (int i(0); i < m_zCount; i++)
	{
		m_disparities[i] = a + i * step;
	}

	// Initialise the restored images and cost
	m_img = cv::UMat::zeros(m_invInd1.size(), CV_8UC3);
	m_imgRectified = cv::UMat::zeros(m_tformInd1.size(), CV_8UC3);
	m_reconsImg = cv::UMat::zeros(m_invInd1.size(), CV_8UC3);
	m_reconsImgRectified = cv::UMat::zeros(m_tformInd1.size(), CV_8UC3);
	m_translatedImg = cv::UMat::zeros(m_tformInd1.size(), CV_8UC3);
	m_reconsImgCandidate = cv::UMat::zeros(m_tformInd1.size(), CV_8UC3);

	m_cost = cv::UMat::zeros(m_tformInd1.size(), CV_8UC1);
	m_costHandle = cv::UMat::zeros(m_tformInd1.size(), CV_8UC1);
	m_costrgb1 = cv::UMat::zeros(m_tformInd1.size(), CV_8UC3);
	m_costrgb2 = cv::UMat::zeros(m_tformInd1.size(), CV_8UC3);
	m_minCost = cv::UMat::zeros(m_tformInd1.size(), CV_8UC1);
	m_maxCost = cv::UMat::zeros(m_tformInd1.size(), CV_8UC1);
	m_maskBest = cv::UMat::zeros(m_tformInd1.size(), CV_8UC1);

	m_fullDisparityMap = cv::UMat::zeros(m_tformInd1.size(), CV_8UC1);
	m_sparseDisparityMap = cv::UMat::zeros(m_invIndMask1.size(), CV_8UC1);
	m_fullDisparityMapConf = cv::UMat::zeros(m_invIndMask1.size(), CV_8UC1);

	m_confidence = cv::UMat::zeros(m_invIndMask1.size(), CV_8UC1);
	m_reconsImgConf = cv::UMat::zeros(m_invIndMask1.size(), CV_8UC3);
	m_minCostConf = cv::UMat::zeros(m_invIndMask1.size(), CV_8UC1);
	m_edges1Conf = cv::UMat::zeros(m_invIndMask1.size(), CV_8UC3);
	m_edges2Conf = cv::UMat::zeros(m_invIndMask1.size(), CV_8UC3);
	m_edgesGreyConf = cv::UMat::zeros(m_invIndMask1.size(), CV_8UC1);
	m_ConfHandle = cv::UMat::zeros(m_invIndMask1.size(), CV_8UC1);
	m_maskConfidence = cv::UMat::zeros(m_invIndMask1.size(), CV_8UC1);

	/// Bilateral filter with confidence map
	cv::ocl::Context context;
	if (!context.create(cv::ocl::Device::TYPE_GPU))
	{
		std::cout << "Failed creating the context, depth filtering will be skipped" << std::endl;
		m_disparityFiltering = false;
	}
	else
	{
		readAndCompileFilter(context);
	}
}

void DepthEstimator::restoreImage(float disparity, float tauLocal, cv::UMat const & imgRectified, 
	cv::UMat & translatedImg, cv::UMat & reconsImgCandidate)
{
	imgRectified.copyTo(reconsImgCandidate);

	for (int k(0); k < 2; k++)
	{
		// Translate the image  
		if (disparity < 0)
		{
			int d(disparity - 0.5);
			reconsImgCandidate(cv::Rect(-d, 0, reconsImgCandidate.cols + d, reconsImgCandidate.rows))
				.copyTo(translatedImg(cv::Rect(0, 0, reconsImgCandidate.cols + d, reconsImgCandidate.rows)));
		}
		else
		{
			int d(disparity + 0.5);
			reconsImgCandidate(cv::Rect(0, 0, reconsImgCandidate.cols - d, reconsImgCandidate.rows))
				.copyTo(translatedImg(cv::Rect(d, 0, reconsImgCandidate.cols - d, reconsImgCandidate.rows)));
		}

		// Multiply by tau
		cv::multiply(translatedImg, tauLocal, translatedImg, 1., CV_8UC3);

		// Reconstruct
		if (k == 0)
			cv::subtract(reconsImgCandidate, translatedImg, reconsImgCandidate);
		else
			cv::add(reconsImgCandidate, translatedImg, reconsImgCandidate);
		// Update parameters
		disparity *= 2.f;
		tauLocal *= tauLocal;
	}
}

void DepthEstimator::readAndCompileFilter(cv::ocl::Context &context)
{
	float sigmaGuide(20.f), guideCoeff(-0.5f / (sigmaGuide*sigmaGuide));
	float sigmaSpace(5.f), gaussSpaceCoeff = -0.5 / (sigmaSpace*sigmaSpace);
	std::vector<float> space_weight(m_filterSize * m_filterSize);
	std::vector<int> space_ofs1(m_filterSize * m_filterSize), space_ofs3(m_filterSize * m_filterSize);

	// Fill-in the filter and indices
	int index = 0;
	for (int i = -m_filterRadius; i <= m_filterRadius; i++)
	{
		for (int j = -m_filterRadius; j <= m_filterRadius; j++)
		{
			float r = std::sqrt((float)i * i + (float)j * j);
			if (r > m_filterRadius)
				continue;
			space_weight[index] = (float)std::exp(r * r * gaussSpaceCoeff);
			space_ofs3[index] = (int)((i * m_sparseDisparityMap.step + j) * 3);
			space_ofs1[index++] = (int)(i * m_sparseDisparityMap.step + j);
		}
	}

	// Create the kernel and index matrices
	cv::Mat(1, index, CV_32FC1, &space_weight[0]).copyTo(m_spaceWeight);
	cv::Mat(1, index, CV_32SC1, &space_ofs1[0]).copyTo(m_filterIndCn1);
	cv::Mat(1, index, CV_32SC1, &space_ofs3[0]).copyTo(m_filterIndCn3);

	// Read ocl code
	std::ifstream ifs("bilateral_filter.cl");
	std::string kernelSourceBilateral((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	cv::ocl::ProgramSource programSourceBilateral(kernelSourceBilateral);

	// Compile the kernel code
	cv::String errmsg;
	cv::ocl::Program programBilateral = context.getProg(programSourceBilateral,
		" -D FILTER_SIZE=" + std::to_string(index)
		+ " -D RADIUS=" + std::to_string(m_filterRadius)
		+ " -D GUIDE_COEFF=" + std::to_string(guideCoeff), errmsg);
	m_kernelBilateral = cv::ocl::Kernel("bilateralFilter", programBilateral);
	std::cout << errmsg;
}

void DepthEstimator::reconstructDepthAndColour()
{
	for (int zInd(0); zInd < m_zCount; zInd++)
	{
		// Reconstruction for each depth candidates
		restoreImage(m_disparities[zInd], m_tau, m_imgRectified, m_translatedImg, m_reconsImgCandidate);

		// Cost computation
		cv::filter2D(m_reconsImgCandidate, m_costrgb1, -1, m_kernelGrad1);
		cv::filter2D(m_reconsImgCandidate, m_costrgb2, -1, m_kernelGrad2);

		cv::add(m_costrgb2, m_costrgb1, m_costrgb1);
		cv::cvtColor(m_costrgb1, m_cost, cv::COLOR_RGB2GRAY);

		cv::boxFilter(m_cost, m_costHandle, -1, cv::Size(m_winSize, 1));
		cv::boxFilter(m_costHandle, m_cost, -1, cv::Size(1, m_winSize));

		// Depth selection and reconstruction merging
		if (zInd == 0)
		{
			m_cost.copyTo(m_minCost);
			m_cost.copyTo(m_maxCost);

			m_fullDisparityMap.setTo(1);
			m_reconsImgCandidate.copyTo(m_reconsImgRectified);
		}
		else
		{
			// Get best depth and update masks
			cv::compare(m_minCost, m_cost, m_maskBest, cv::CMP_GE);
			m_cost.copyTo(m_minCost, m_maskBest);

			cv::max(m_maxCost, m_cost, m_maxCost);
			m_fullDisparityMap.setTo(zInd + 1, m_maskBest);

			// Merge reconstructions
			cv::copyTo(m_reconsImgCandidate, m_reconsImgRectified, m_maskBest);
		}
	}
}

void DepthEstimator::unwarpAndFixColour()
{
	// Account for the intensity drop of the restoration algorithm and of the e-ray removal
	cv::multiply(m_reconsImgRectified, (1.f + m_tau) / (1.f + std::pow(m_tau, 4)), m_reconsImgRectified);

	// Reverse rectification
	cv::remap(m_reconsImgRectified, m_reconsImg, m_invInd1, m_invInd2, cv::INTER_LINEAR);
	cv::remap(m_reconsImgRectified, m_reconsImgConf, m_invIndMask1, m_invIndMask2, cv::INTER_LINEAR);
	cv::remap(m_fullDisparityMap, m_fullDisparityMapConf, m_invIndMask1, cv::noArray(), cv::INTER_NEAREST);
	cv::remap(m_maxCost, m_confidence, m_invIndMask1, cv::noArray(), cv::INTER_NEAREST);
	cv::remap(m_minCost, m_minCostConf, m_invIndMask1, cv::noArray(), cv::INTER_NEAREST);

	// Use the original image at the boundary as our restoration cannot handle those areas
	m_img(cv::Rect(0, 0, m_img.cols, 5)).copyTo(m_reconsImg(cv::Rect(0, 0, m_img.cols, 5)));
	m_img(cv::Rect(0, m_img.rows - 5, m_img.cols, 5))
		.copyTo(m_reconsImg(cv::Rect(0, m_img.rows - 5, m_img.cols, 5)));
	m_img(cv::Rect(m_img.cols - 40, 0, 40, m_img.rows))
		.copyTo(m_reconsImg(cv::Rect(m_img.cols - 40, 0, 40, m_img.rows)));
}

void DepthEstimator::maskDisparityMap()
{	
	// Build the confidence map using the difference between the best and worse cost
	cv::subtract(m_confidence, m_minCostConf, m_minCostConf);
	cv::compare(m_minCostConf, m_threshCost, m_maskConfidence, cv::CMP_LE);

	// Map displacement to account for the position of the artefacts 
	// when the image is reconstructed with a wrong depth candidate
	int displacement = int(float(m_winSize * m_fullDisparityMapConf.cols) / (m_fullDisparityMap.cols  * 2));
	m_fullDisparityMapConf.copyTo(m_ConfHandle);
	m_ConfHandle(cv::Rect(0, 0, m_ConfHandle.cols - displacement, m_ConfHandle.rows))
		.copyTo(m_fullDisparityMapConf(cv::Rect(displacement, 0, m_ConfHandle.cols - displacement, m_ConfHandle.rows)));
	m_confidence.copyTo(m_ConfHandle);
	m_ConfHandle(cv::Rect(0, 0, m_ConfHandle.cols - displacement, m_ConfHandle.rows))
		.copyTo(m_confidence(cv::Rect(displacement, 0, m_ConfHandle.cols - displacement, m_ConfHandle.rows)));

	// Create the mask
	cv::add(m_confidence, -1, m_confidence);
	m_confidence.setTo(0, m_maskConfidence);
	m_confidence.setTo(1, m_confidence);

	// Refine the confidence map using the edge structure in the restored image
	cv::filter2D(m_reconsImgConf, m_edges1Conf, -1, m_kernelGrad1);
	cv::filter2D(m_reconsImgConf, m_edges2Conf, -1, m_kernelGrad2);
	cv::add(m_edges2Conf, m_edges1Conf, m_edges1Conf);
	cv::cvtColor(m_edges1Conf, m_edgesGreyConf, cv::COLOR_RGB2GRAY);
	cv::compare(m_edgesGreyConf, m_threshGrad, m_maskConfidence, cv::CMP_LT);
	m_confidence.setTo(0, m_maskConfidence);
	cv::erode(m_confidence, m_confidence, cv::UMat::ones(2, 2, CV_8UC1));
	
	cv::compare(m_confidence, 0, m_maskConfidence, cv::CMP_EQ);
	m_fullDisparityMapConf.setTo(0, m_maskConfidence);
}

void DepthEstimator::filterDisparity()
{
	if (m_disparityFiltering)
	{
		// Run filter
		size_t globalThreads[2] = { size_t(m_fullDisparityMapConf.cols), size_t(m_fullDisparityMapConf.rows) };
		size_t localThreads[2] = { 32, 32 };

		cv::multiply(m_fullDisparityMapConf, 255. / m_zCount, m_fullDisparityMapConf);
		m_sparseDisparityMap.setTo(0);

		m_kernelBilateral.args(
			cv::ocl::KernelArg::ReadOnlyNoSize(m_fullDisparityMapConf),
			cv::ocl::KernelArg::ReadOnlyNoSize(m_reconsImgConf),
			cv::ocl::KernelArg::WriteOnly(m_sparseDisparityMap),
			m_spaceWeight.handle(cv::ACCESS_READ),
			m_filterIndCn1.handle(cv::ACCESS_READ), m_filterIndCn3.handle(cv::ACCESS_READ)
		);
		m_kernelBilateral.run(2, globalThreads, localThreads, true);

		// Outlier removal
		cv::absdiff(m_fullDisparityMapConf, m_sparseDisparityMap, m_fullDisparityMapConf);
		cv::compare(m_fullDisparityMapConf, 6, m_maskConfidence, cv::CMP_GT);
		m_sparseDisparityMap.setTo(0, m_maskConfidence);
	}
	else
	{
		cv::multiply(m_fullDisparityMapConf, 255. / m_zCount, m_sparseDisparityMap);
	}
}