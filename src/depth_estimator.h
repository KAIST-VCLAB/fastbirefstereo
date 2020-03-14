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

#ifndef DEPTHESTIMATOR_H
#define DEPTHESTIMATOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

/* @class  DepthEstimator
@brief  DepthEstimator is a class to estimate the depth 
and reconstruct the image for uneven birefractive stereo.
*/
class DepthEstimator
{
public:
	/* @brief Set parameters, read LuTs and initialise variables
	@param tformInd of the rectification remapping table
	@param invInd the table to reverse rectification
	@param minZ Lowest depth candidate
	@param maxZ Largest depth candidate
	@param disparityCoef f * baseline such as disparity_{o->e} = disparityCoef * 1 / depth
	in the horizontal direction. 
	@param tau intensity proportion between e-ray and o-ray: I_captured = tau * I_e + I_o, 0 < tau < 1
	@param upsampling upsampling parameter to increase accuracy at the cost of memory and runtime
	@param scaleMask Resize te disparity map before masking to improve performance
	@param winSize Window size for cost computation
	@param threshGrad Mask out in the disparity map areas with lower gradient in the reconstructed image
	@param threshCost Mask out in the disparity map areas with lower cost difference between the minimum and maximum
	*/
	DepthEstimator(cv::UMat const & tformInd, cv::UMat const & invInd,
		float minZ, float maxZ, float disparityCoef, float tau,
		float upsampling = 1.f, double scaleMask = 0.3,
		int winSize = 61, unsigned char threshGrad = 220, unsigned char threshCost = 1);

	/* @brief set a new uneven birefractive image and run the restoration algorithm
	@param img uneven birefractive image (CV_8UC3)
	*/
	inline void setFrame(const cv::UMat & img);

	/* @brief Restore a rectified birefractive image for a given disparity and tau value
	@param disparity disparity candidate between e-ray and o-ray
	@param tau intensity proportion in uneven double refraction (I_captured = tau * I_e + I_o, 0 < tau < 1)
	@param imgRectified Rectified uneven birefractive image (CV_8UC3)
	@param translatedImg Handle for image translation. Must be initialised 
	and have the same size as imgRectified (CV_8UC3)
	@param reconsImgCandidate Output restored image (CV_8UC3)
	*/
	static void restoreImage(float disparity, float tau, cv::UMat const & imgRectified, cv::UMat & translatedImg, cv::UMat & reconsImgCandidate);

	/* @brief Convert the disparity map computed in setFrame to depth 
	@return depth map in mm (CV_32FC1)
	*/
	inline const cv::UMat getDepth();

	/* @brief Get the coloured disparity map after being computed in setFrame
	@return coloured disparity map with cv::COLORMAP_MAGMA (CV_8UC3)
	*/
	inline const cv::UMat getDisparityMap();

	/* @brief Get the restored image after being computed in setFrame
	@return restored image (CV_8UC3)
	*/
	inline const cv::UMat getReconsImg();

private:
	/* Compile "bilateral_filter.cl" code for disparity map filtering */
	void readAndCompileFilter(cv::ocl::Context &context);

	/* RestoreImage for all depth candidate, 
	compute cost and select the best depth and colour */
	void reconstructDepthAndColour();

	/* Reverse rectification and tweak the colour image 
	fix intensity and boundaries
	*/
	void unwarpAndFixColour();

	/* Compute confidence and mask out unreliable areas in the disparity map */
	void maskDisparityMap();

	/* Filter the sparse disparity map using a bilateral filter */
	void filterDisparity();

	/// Parameters
	// Horizontal f*baseline: disparity = m_disparityCoef / depth
	float m_disparityCoef;
	float m_tau; // intensity proportion between e-ray and o-ray
	int m_winSize; // Window for cost computation
	int m_zCount; // Number of depth candidates
	std::vector<float> m_disparities; // disparity candidates
	unsigned char m_threshGrad; // Threshold for vertical edges in mask computation
	unsigned char m_threshCost; // Threshold for clear winner in mask computation
	
	// Rectification tables
	cv::UMat m_tformInd1, m_tformInd2, 
		m_invInd1, m_invInd2, m_invIndMask1, m_invIndMask2;

	/// Image and colour restoration
	cv::UMat m_img;
	cv::UMat m_imgRectified; // Rectified input image
	cv::UMat m_reconsImg; // Restored image
	cv::UMat m_reconsImgRectified; // Rectified restored image
	cv::UMat m_translatedImg; // Translated image handler for image reconstruction
	cv::UMat m_reconsImgCandidate; // Restored image for a given candidate
	
	/// Cost computation
	// Cost and handles for a given candidate
	cv::UMat m_cost, m_costHandle, m_costrgb1, m_costrgb2;
	// Best and worse cost for candidate selection and mask computation
	cv::UMat m_minCost, m_maxCost;
	cv::UMat m_maskBest; // Mask of where the current candidate is the best
	// filters for gradient computation
	cv::Mat m_kernelGrad1, m_kernelGrad2;

	/// Disparity maps
	// Disparity map after winner-takes all on all pixels
	cv::UMat m_fullDisparityMap; 
	// Disparity map with unreliable areas filtered out
	cv::UMat m_sparseDisparityMap;
	// Resized disparity map for confidence estimation
	cv::UMat m_fullDisparityMapConf;

	/// Mask computation
	cv::UMat m_confidence; // Confidence map for reliable areas
	cv::UMat m_reconsImgConf; // Resized restored image
	cv::UMat m_minCostConf; // Resized best cost
	// Edges in the restored image for reliable area estimation
	cv::UMat m_edges1Conf, m_edges2Conf, m_edgesGreyConf; 
	// Handle for some conputations on the confidence
	cv::UMat m_ConfHandle, m_maskConfidence;

	/// Disparity map filtering 
	static const int m_filterSize = 21, m_filterRadius = m_filterSize / 2;
	bool m_disparityFiltering = true;
	cv::ocl::Kernel m_kernelBilateral; // ocl kernel for disparity filtering
	// weights and indices for disparity map filtering
	cv::UMat m_spaceWeight, m_filterIndCn1, m_filterIndCn3;
};


inline void DepthEstimator::setFrame(const cv::UMat & img)
{
	m_img = img;
	cv::remap(m_img, m_imgRectified, m_tformInd1, m_tformInd2, cv::INTER_LINEAR);
	reconstructDepthAndColour();
	unwarpAndFixColour();
	maskDisparityMap();
	filterDisparity();
}

inline const cv::UMat DepthEstimator::getDepth()
{
	cv::UMat depth;
	
	// Map disparity to the [0, 1] range
	m_sparseDisparityMap.convertTo(depth, CV_32F, 
		1. / (255.), -1. / m_zCount);
	// Map to the [1. / maxDepth, 1. / minDepth] range
	double minDepth(1. / (m_disparities[m_zCount - 1] / m_disparityCoef)), maxDepth(1. / (m_disparities[0] / m_disparityCoef));
	cv::multiply(depth, (1. / minDepth - 1. / maxDepth), depth);
	cv::add(depth, 1. / maxDepth, depth);
	// Convert to depth
	cv::divide(1., depth, depth);
	// Mask out unreliable areas
	cv::compare(m_sparseDisparityMap, 0, m_maskConfidence, cv::CMP_EQ);
	depth.setTo(0., m_maskConfidence);

	return depth;
}

inline const cv::UMat DepthEstimator::getDisparityMap()
{
	cv::UMat disparityMap;

	// Shift the disparity map to get a better sparse visualisation
	cv::multiply(m_sparseDisparityMap, 0.8, disparityMap);
	cv::add(disparityMap, 0.25 * 255., disparityMap, disparityMap);
	cv::dilate(disparityMap, disparityMap, cv::UMat::ones(3, 3, CV_8UC1));
	cv::applyColorMap(disparityMap, disparityMap, cv::COLORMAP_MAGMA);

	return disparityMap;
}

inline const cv::UMat DepthEstimator::getReconsImg()
{
	return m_reconsImg;
}
#endif // DEPTHESTIMATOR_H