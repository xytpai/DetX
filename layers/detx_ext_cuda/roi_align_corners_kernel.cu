#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include "detx_common.h"


__global__ void roi_align_corners_forward_kernel(const int nthreads, 
	const float *features, const float *rois, float *output, 
	const int c, const int ph, const int pw, const int n, 
	const int feature_stride, const int out_size)
{
	CUDA_1D_KERNEL_LOOP(i, nthreads) {
		int ow_i = i % out_size;
		int oh_i = (i / out_size) % out_size;
		int c_i = (i / out_size / out_size) % c;
		int n_i = i / c / out_size / out_size;
		float ymin = rois[n_i*4+0];
		float xmin = rois[n_i*4+1];
		float ymax = rois[n_i*4+2];
		float xmax = rois[n_i*4+3];
		float fy = (ymin + oh_i*(ymax - ymin) / (out_size-1)) / feature_stride;
		float fx = (xmin + ow_i*(xmax - xmin) / (out_size-1)) / feature_stride;
		int tl_y = (int)fy, tl_x = (int)fx;
        int tr_y = tl_y,    tr_x = tl_x+1;
        int bl_y = tl_y+1,  bl_x = tl_x;
		int br_y = tl_y+1,  br_x = tl_x+1;
		if (tl_y<0 || tl_x<0 || br_y>=ph || br_x>=pw) continue;
		int base = c_i*ph*pw;
		// float tl = (tl_y>=0 && tl_x>=0) ? features[base + tl_y*pw + tl_x] : 0;
        // float tr = (tr_y>=0 && tr_x<pw) ? features[base + tr_y*pw + tr_x] : 0;
        // float bl = (bl_y<ph && bl_x>=0) ? features[base + bl_y*pw + bl_x] : 0;
		// float br = (br_y<ph && br_x<pw) ? features[base + br_y*pw + br_x] : 0;
		float tl = features[base + tl_y*pw + tl_x];
		float tr = features[base + tr_y*pw + tr_x];
		float bl = features[base + bl_y*pw + bl_x];
		float br = features[base + br_y*pw + br_x];
		float dy = fy - (float)tl_y;
		float dx = fx - (float)tl_x;
		float score = tl*(1.0-dy)*(1.0-dx) + tr*(1.0-dy)*dx + bl*dy*(1.0-dx) + br*dy*dx;
		output[n_i*c*out_size*out_size+c_i*out_size*out_size+oh_i*out_size+ow_i] = score;
	}
}
at::Tensor roi_align_corners_forward_cuda(
	const at::Tensor &features, const at::Tensor &rois, 
    const int feature_stride, const int out_size)
{
	// features: F(c, ph, pw)
	// rois: F(n, 4) ymin, xmin, ymax, xmax
	// return: F(n, c, out_size, out_size)
	const int c  = features.size(0);
	const int ph = features.size(1);
	const int pw = features.size(2);
	const int n  = rois.size(0);
	auto output = at::zeros({n, c, out_size, out_size}, features.options());
	if (output.numel() == 0) {
		THCudaCheck(cudaGetLastError());
		return output;
	}
	const int nthreads = n*c*out_size*out_size;
	dim3 grid(std::min(DivUp(nthreads, MAX_BLOCK_SIZE), MAX_GRID_SIZE)), block(MAX_BLOCK_SIZE);
	roi_align_corners_forward_kernel<<<grid, block>>>(nthreads, 
		features.contiguous().data<float>(),
		rois.contiguous().data<float>(),
		output.contiguous().data<float>(),
		c, ph, pw, n, feature_stride, out_size);
	THCudaCheck(cudaGetLastError());
	return output;
}



__global__ void roi_align_corners_backward_kernel(const int nthreads, 
	const float *d_losses, const float *rois, float *output, 
	const int c, const int ph, const int pw, const int n, 
	const int feature_stride, const int out_size)
{
	CUDA_1D_KERNEL_LOOP(i, nthreads) {
		int ow_i = i % out_size;
		int oh_i = (i / out_size) % out_size;
		int c_i = (i / out_size / out_size) % c;
		int n_i = i / c / out_size / out_size;
		float ymin = rois[n_i*4+0];
		float xmin = rois[n_i*4+1];
		float ymax = rois[n_i*4+2];
		float xmax = rois[n_i*4+3];
		float fy = (ymin + oh_i*(ymax - ymin) / (out_size-1)) / feature_stride;
		float fx = (xmin + ow_i*(xmax - xmin) / (out_size-1)) / feature_stride;
		int tl_y = (int)fy, tl_x = (int)fx;
        int tr_y = tl_y,    tr_x = tl_x+1;
        int bl_y = tl_y+1,  bl_x = tl_x;
		int br_y = tl_y+1,  br_x = tl_x+1;
		if (tl_y<0 || tl_x<0 || br_y>=ph || br_x>=pw) continue;
		int base = c_i*ph*pw;
		float dy = fy - (float)tl_y;
		float dx = fx - (float)tl_x;
		float t = d_losses[n_i*c*out_size*out_size+c_i*out_size*out_size+oh_i*out_size+ow_i];
		// if (tl_y>=0 && tl_x>=0)
		atomicAdd(&output[base + tl_y*pw + tl_x], (1.0-dy)*(1.0-dx)*t);
		// if (tr_y>=0 && tr_x<pw)
		atomicAdd(&output[base + tr_y*pw + tr_x], (1.0-dy)*dx*t);
		// if (bl_y<ph && bl_x>=0)
		atomicAdd(&output[base + bl_y*pw + bl_x], dy*(1.0-dx)*t);
		// if (br_y<ph && br_x<pw)
		atomicAdd(&output[base + br_y*pw + br_x], dy*dx*t);
	}
}
at::Tensor roi_align_corners_backward_cuda(
	const at::Tensor &d_losses, const at::Tensor &rois, 
	const int ph, const int pw, 
    const int feature_stride, const int out_size)
{
	// d_losses: F(n, c, out_size, out_size)
	// rois: F(n, 4) ymin, xmin, ymax, xmax
	// return: F(c, ph, pw)
	const int c  = d_losses.size(1);
	const int n  = rois.size(0);
	auto output = at::zeros({c, ph, pw}, d_losses.options());
	if (output.numel() == 0) {
		THCudaCheck(cudaGetLastError());
		return output;
	}
	const int nthreads = n*c*out_size*out_size;
	dim3 grid(std::min(DivUp(nthreads, MAX_BLOCK_SIZE), MAX_GRID_SIZE)), block(MAX_BLOCK_SIZE);
	roi_align_corners_backward_kernel<<<grid, block>>>(nthreads, 
		d_losses.contiguous().data<float>(),
		rois.contiguous().data<float>(),
		output.contiguous().data<float>(),
		c, ph, pw, n, feature_stride, out_size);
	THCudaCheck(cudaGetLastError());
	return output;
}
