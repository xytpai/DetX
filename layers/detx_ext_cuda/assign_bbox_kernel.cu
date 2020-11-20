#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include "detx_common.h"


__global__ void assign_box_kernel(const int nthreads, 
	const float *box, long *target, const int ph, const int pw, const int stride, 
	const float size_min, const float size_max, const float radius, const int n)
{
	float size_min_2 = size_min/2.0;
	float size_max_2 = size_max/2.0;
	float _radius = radius*stride;
	CUDA_1D_KERNEL_LOOP(i, nthreads) {
		int pw_i = i % pw;
		int ph_i = i / pw;
		float center_y = ph_i * stride;
		float center_x = pw_i * stride;
		int res = -1, box_base = i*n*4;
		float min_area = 999999999;
		for (int n_i=0; n_i<n; ++n_i) {
			float ymin = box[box_base+0];
			float xmin = box[box_base+1];
			float ymax = box[box_base+2];
			float xmax = box[box_base+3];
			box_base += 4;
			float cy = (ymin + ymax) / 2.0;
			float cx = (xmin + xmax) / 2.0;
			float ch = ymax - ymin + 1;
			float cw = xmax - xmin + 1;
			float area = ch * cw;
			float top = center_y - ymin;
			float bottom = ymax - center_y;
			float left = center_x - xmin;
			float right = xmax - center_x;
			float max_tlbr = max(top, max(left, max(bottom, right)));
			float oy = fabs(cy - center_y);
			float ox = fabs(cx - center_x);
			if (max_tlbr>=size_min_2 && max_tlbr<=size_max_2 
				&& (top > 0) && (bottom > 0) && (left > 0) && (right > 0) 
				&& oy<=_radius && ox<=_radius && area<=min_area) {
				res = n_i;
				min_area = area;
			}
		}
		target[i] = res;
	}
}
void assign_box_cuda(
	const at::Tensor &box, const int stride,
	const float size_min, const float size_max, const float radius, 
	at::Tensor &target)
{
	// box: F(ph, pw, n, 4) ymin, xmin, ymax, xmax
	// ->target: L(ph, pw) 0~n-1, bg:-1
	const int ph = box.size(0);
	const int pw = box.size(1);
	const int n  = box.size(2);
	const int nthreads = ph*pw;
	dim3 grid(std::min(DivUp(nthreads, MAX_BLOCK_SIZE), MAX_GRID_SIZE)), block(MAX_BLOCK_SIZE);
	assign_box_kernel<<<grid, block>>>(nthreads, 
		box.contiguous().data<float>(),
		target.contiguous().data<long>(),
		ph, pw, stride, size_min, size_max, radius, n);
	THCudaCheck(cudaGetLastError());
}
