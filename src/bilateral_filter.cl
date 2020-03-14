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

#define SUM(a) a.x + a.y + a.z
#define loadpix3(addr) vload3(0, (__global const uchar *)(addr))

/*
@brief Simple bilateral filter for sparse disparity maps
@param src input sparse disparity map
@param guide colour image to guide filtering
@param dst output sparse filtered disparity map
@param space_weight gaussian spatial weights
@param space_ofs1 index offset for one channel (disparity maps)
@param space_ofs3 index offset for three channels (guide)
*/
__kernel void bilateralFilter(__global const uchar * src, int src_step, int src_offset,
	__global const uchar * guide, int guide_step, int guide_offset,
	__global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
	__constant float * space_weight, __constant int * space_ofs1, __constant int * space_ofs3)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	if(x > RADIUS && y > RADIUS && x < dst_cols - RADIUS && y < dst_rows - RADIUS)
	{
		long int src_index = mad24(y, src_step, x + src_offset);
		long int guide_index = guide + mad24(y, guide_step, mad24(x, 3, guide_offset));
		long int dst_index = mad24(y, dst_step, x + dst_offset);

		// Filter if the disparity is not 0
		if (src[src_index])
		{
			float sum = 0.f;
			float wsum = 0.f;
			short3 val0 = convert_short3(loadpix3(guide_index));
			// Aggregate over non 0 neighbour pixels
			for (int k = 0; k < FILTER_SIZE; k++)
			{
				short3 val = convert_short3(loadpix3(guide_index + space_ofs3[k]));
				short diff = SUM(abs(val - val0)); // Colour difference between the two points
				uchar src_val = src[src_index + space_ofs1[k]];
				// Compute sparsity-aware bilateral weight
				float w = 
					(src_val != 0) // Ignore 0 neighbour pixels
					* space_weight[k] // Gaussian 2D component
					* native_exp((float)(diff * diff) * GUIDE_COEFF); // Image consistency component
				sum += (float)(src_val)* w;
				wsum += w;
			}
			dst[dst_index] = (uchar)(sum / wsum);
		}
	}
}
