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

#ifndef RECTIFIER_H
#define RECTIFIER_H

#include <opencv2/core/core.hpp>

/* @class Rectifier
@brief Class to handle uneven double refraction rectification for RGB-D imaging
*/
class Rectifier
{
public:
	/* @brief Build rectification mapping tables and their inverts via dynamic programming.
	Type should be CV_32F2 for all input and outputs
	@param bO2D baseline from o-ray to d-ray (given by Baek et al.'s code)
	@param bE2D the baseline from e-ray to d-ray (given by Baek et al.'s code)
	@param tformInd component of the rectification remapping table
	@param invInd component of the table to reverse rectification
	*/
	static float buildRectification(cv::UMat const & bO2D, cv::UMat const & bE2D,
		cv::UMat & tformInd, cv::UMat & invInd);

	/* @brief Generic function to reverse continuous remapping.
	Does not use explicit nearest neighbours search
	as the local consistency of the rectification makes it possible without
	@param tformInd remapping to reverse (CV_32FC2)
	@param invInd output inverse. Needs to be initialised to the desired size (CV_32FC2)
	@param scale for smoothness. Prevents artifacts as no interpolation is used
	*/
	static void reverseRectification(cv::UMat const & tformInd, 
		cv::UMat & invInd, double scale = 6.);

private:
	Rectifier() {}
};
#endif // RECTIFIER_H