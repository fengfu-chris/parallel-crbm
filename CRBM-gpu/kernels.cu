#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include "./headers/defines.h"

__global__ void conv3d_valid_kernel(float *d_H, float *d_V, float *d_W, int nv, int nw)
{   
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // load W to shared memory
    __shared__ float Ws[WIDTH][WIDTH];

    if(ty < nw && tx < nw){        
        Ws[ty][tx] = d_W[nw*nw*gridDim.x*by + nw*nw*bx + nw*ty + tx];
    }
    __syncthreads();

    const int nh = nv - nw + 1;
    const int tiles_x = nh / blockDim.x + 1;
	const int tiles_y = nh / blockDim.y + 1;

    for(int iy=0; iy < tiles_y; iy++){
        for(int ix=0; ix < tiles_x; ix++){     
            int idH_y = ty + iy * blockDim.y;
            int idH_x = tx + ix * blockDim.x;     

            if(idH_x >= nh || idH_y >= nh)
                continue;

            float temp = 0.0;
            for(int k=0; k<nw; k++){
                for(int l=0; l<nw; l++){
                    int idV_y = idH_y + k;
                    int idV_x = idH_x + l;
                    float v = d_V[nv*nv*bx + nv*idV_y + idV_x];
                    float w = Ws[nw-1-k][nw-1-l];
                    temp += v*w;
                }
            }
            atomicAdd(&d_H[nh*nh*by + idH_y*nh + idH_x], temp);
        }
    }
}

__global__ void conv4d_valid_kernel(float *d_H, float *d_V, float *d_W, int nv, int nw)
{   
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
	const int bz = blockIdx.z;

    // load W to shared memory
    __shared__ float Ws[WIDTH][WIDTH];

    if(ty < nw && tx < nw){        
        Ws[ty][tx] = d_W[nw*nw*gridDim.x*by + nw*nw*bx + nw*ty + tx];
    }
    __syncthreads();

    const int nh = nv - nw + 1;
    const int tiles_x = nh / blockDim.x + 1;
	const int tiles_y = nh / blockDim.y + 1;

    for(int iy=0; iy < tiles_y; iy++){
        for(int ix=0; ix < tiles_x; ix++){     
            int idH_y = ty + iy * blockDim.y;
            int idH_x = tx + ix * blockDim.x;     

            if(idH_x >= nh || idH_y >= nh)
                continue;

            float temp = 0.0;
            for(int k=0; k<nw; k++){
                for(int l=0; l<nw; l++){
                    int idV_y = idH_y + k;
                    int idV_x = idH_x + l;
                    float v = d_V[nv*nv*gridDim.x*bz + nv*nv*bx + nv*idV_y + idV_x];
                    float w = Ws[nw-1-k][nw-1-l];
                    temp += v*w;
                }
            }

            atomicAdd(&d_H[nh*nh*gridDim.y*bz + nh*nh*by + idH_y*nh + idH_x], temp);
        }
    }
}

__global__ void conv3d_full_kernel(float *d_V, float *d_H, float *d_W, int nh, int nw)
{   
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // load W to shared memory
    __shared__ float Ws[WIDTH][WIDTH];

    if(ty < nw && tx < nw){        
        Ws[ty][tx] = d_W[nw*nw*gridDim.x*by + nw*nw*bx + nw*ty + tx];
    }
    __syncthreads();

    const int nv = nh + nw - 1;
    const int tiles_x = nv / blockDim.x + 1;
	const int tiles_y = nv / blockDim.y + 1;

    for(int iy=0; iy < tiles_y; iy++){
        for(int ix=0; ix < tiles_x; ix++){     
            int idV_y = ty + iy * blockDim.y;
            int idV_x = tx + ix * blockDim.x;     

            if(idV_x >= nv || idV_y >= nv)
                continue;

            float temp = 0.0;
            for(int k=0; k<nw; k++){
                for(int l=0; l<nw; l++){
   					int idH_y = idV_y + k - nw + 1;
                    int idH_x = idV_x + l - nw + 1;
                    if(idH_y < 0 || idH_x < 0 || idH_y >= nh || idH_x >= nh) 
                        continue;

					float h = d_H[nh*nh*by + idH_y*nh + idH_x];
					float w = Ws[nw-1-k][nw-1-l];
                    temp += h*w;
                }
            }
            atomicAdd(&d_V[nv*nv*bx + idV_y*nv + idV_x], temp);
        }
    }
}

__global__ void conv4d_full_kernel(float *d_V, float *d_H, float *d_W, int nh, int nw)
{ 
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
	const int bz = blockIdx.z;

    // load W to shared memory
    __shared__ float Ws[WIDTH][WIDTH];

    if(ty < nw && tx < nw){        
        Ws[ty][tx] = d_W[nw*nw*gridDim.x*by + nw*nw*bx + nw*ty + tx];
    }
    __syncthreads();

    const int nv = nh + nw - 1;
    const int tiles_x = nv / blockDim.x + 1;
	const int tiles_y = nv / blockDim.y + 1;

    for(int iy=0; iy < tiles_y; iy++){
        for(int ix=0; ix < tiles_x; ix++){     
            int idV_y = ty + iy * blockDim.y;
            int idV_x = tx + ix * blockDim.x;     

            if(idV_x >= nv || idV_y >= nv)
                continue;

            float temp = 0.0;
            for(int k=0; k<nw; k++){
                for(int l=0; l<nw; l++){
   					int idH_y = idV_y + k - nw + 1;
                    int idH_x = idV_x + l - nw + 1;
                    if(idH_y < 0 || idH_x < 0 || idH_y >= nh || idH_x >= nh) 
                        continue;

					float h = d_H[nh*nh*gridDim.y*bz + nh*nh*by + idH_y*nh + idH_x];
					float w = Ws[nw-1-k][nw-1-l];
                    temp += h*w;
                }
            }
            atomicAdd(&d_V[nv*nv*gridDim.x*bz + nv*nv*bx + idV_y*nv + idV_x], temp);
        }
    }
}

__global__ void filter3d_valid_kernel(float *d_H, float *d_V, float *d_W, int nv, int nw)
{   
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // load W to shared memory
    __shared__ float Ws[WIDTH][WIDTH];

    if(ty < nw && tx < nw){        
        Ws[ty][tx] = d_W[nw*nw*gridDim.x*by + nw*nw*bx + nw*ty + tx];
    }
    __syncthreads();

    const int nh = nv - nw + 1;
    const int tiles_x = nh / blockDim.x + 1;
	const int tiles_y = nh / blockDim.y + 1;

    for(int iy=0; iy < tiles_y; iy++){
        for(int ix=0; ix < tiles_x; ix++){     
            int idH_y = ty + iy * blockDim.y;
            int idH_x = tx + ix * blockDim.x;     

            if(idH_x >= nh || idH_y >= nh)
                continue;

            float temp = 0.0;
            for(int k=0; k<nw; k++){
                for(int l=0; l<nw; l++){
                    int idV_y = idH_y + k;
                    int idV_x = idH_x + l;
                    float v = d_V[nv*nv*bx + nv*idV_y + idV_x];
                    float w = Ws[k][l];
                    temp += v*w;
                }
            }
            atomicAdd(&d_H[nh*nh*by + idH_y*nh + idH_x], temp);
        }
    }
}

__global__ void filter4d_valid_kernel(float *d_H, float *d_V, float *d_W, int nv, int nw)
{   
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
	const int bz = blockIdx.z;

    // load W to shared memory
    __shared__ float Ws[WIDTH][WIDTH];

    if(ty < nw && tx < nw){        
        Ws[ty][tx] = d_W[nw*nw*gridDim.x*by + nw*nw*bx + nw*ty + tx];
    }
    __syncthreads();

    const int nh = nv - nw + 1;
    const int tiles_x = nh / blockDim.x + 1;
	const int tiles_y = nh / blockDim.y + 1;

    for(int iy=0; iy < tiles_y; iy++){
        for(int ix=0; ix < tiles_x; ix++){     
            int idH_y = ty + iy * blockDim.y;
            int idH_x = tx + ix * blockDim.x;     

            if(idH_x >= nh || idH_y >= nh)
                continue;

            float temp = 0.0;
            for(int k=0; k<nw; k++){
                for(int l=0; l<nw; l++){
                    int idV_y = idH_y + k;
                    int idV_x = idH_x + l;
                    float v = d_V[nv*nv*gridDim.x*bz + nv*nv*bx + nv*idV_y + idV_x];
                    float w = Ws[k][l];
                    temp += v*w;
                }
            }
            atomicAdd(&d_H[nh*nh*gridDim.y*bz + nh*nh*by + idH_y*nh + idH_x], temp);
        }
    }
}

__global__ void filter3d_full_kernel(float *d_V, float *d_H, float *d_W, int nh, int nw)
{   
	const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // load W to shared memory
    __shared__ float Ws[WIDTH][WIDTH];

    if(ty < nw && tx < nw){        
        Ws[ty][tx] = d_W[nw*nw*gridDim.x*by + nw*nw*bx + nw*ty + tx];
    }
    __syncthreads();

    const int nv = nh + nw - 1;
    const int tiles_x = nv / blockDim.x + 1;
	const int tiles_y = nv / blockDim.y + 1;

    for(int iy=0; iy < tiles_y; iy++){
        for(int ix=0; ix < tiles_x; ix++){     
            int idV_y = ty + iy * blockDim.y;
            int idV_x = tx + ix * blockDim.x;     

            if(idV_x >= nv || idV_y >= nv)
                continue;

            float temp = 0.0;
            for(int k=0; k<nw; k++){
                for(int l=0; l<nw; l++){
   					int idH_y = idV_y + k - nw + 1;
                    int idH_x = idV_x + l - nw + 1;
                    if(idH_y < 0 || idH_x < 0 || idH_y >= nh || idH_x >= nh) 
                        continue;

					float h = d_H[nh*nh*by + idH_y*nh + idH_x];
					float w = Ws[k][l];
                    temp += h*w;
                }
            }
            atomicAdd(&d_V[nv*nv*bx + idV_y*nv + idV_x], temp);
        }
    }
}


__global__ void filter4d_full_kernel(float *d_V, float *d_H, float *d_W, int nh, int nw)
{   
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
	const int bz = blockIdx.z;

    // load W to shared memory
    __shared__ float Ws[WIDTH][WIDTH];

    if(ty < nw && tx < nw){        
        Ws[ty][tx] = d_W[nw*nw*gridDim.x*by + nw*nw*bx + nw*ty + tx];
    }
    __syncthreads();

    const int nv = nh + nw - 1;
    const int tiles_x = nv / blockDim.x + 1;
	const int tiles_y = nv / blockDim.y + 1;

    for(int iy=0; iy < tiles_y; iy++){
        for(int ix=0; ix < tiles_x; ix++){     
            int idV_y = ty + iy * blockDim.y;
            int idV_x = tx + ix * blockDim.x;     

            if(idV_x >= nv || idV_y >= nv)
                continue;

            float temp = 0.0;
            for(int k=0; k<nw; k++){
                for(int l=0; l<nw; l++){
   					int idH_y = idV_y + k - nw + 1;
                    int idH_x = idV_x + l - nw + 1;
                    if(idH_y < 0 || idH_x < 0 || idH_y >= nh || idH_x >= nh) 
                        continue;

					float h = d_H[nh*nh*gridDim.y*bz + nh*nh*by + idH_y*nh + idH_x];
					float w = Ws[k][l];
                    temp += h*w;
                }
            }
            atomicAdd(&d_V[nv*nv*gridDim.x*bz + nv*nv*bx + idV_y*nv + idV_x], temp);
        }
    }
}

__global__ void mapping_kernel(float *d_H, float *d_hb, int nh)
{   
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int bx = blockIdx.x;  // Kout id
	const int by = blockIdx.y;  // batchsize id

	const int tiles_x = nh / blockDim.x + 1;
	const int tiles_y = nh / blockDim.y + 1;
    for(int ix=0; ix < tiles_x; ix++){
        for(int iy = 0; iy < tiles_y; iy++){           
            int idy = ty + iy * blockDim.y;
            int idx = tx + ix * blockDim.x;
            if(idx >= nh || idy >= nh) continue;

            d_H[nh*nh*gridDim.x*by + nh*nh*bx + nh*idy + idx] += d_hb[bx];
        }
    }
}

__global__ void sampling_kernel(float *d_H, float *d_HS, float *d_r, int C, int nh)
{   
    const int tx = threadIdx.x; 
    const int ty = threadIdx.y;
    const int bx = blockIdx.x; 
    const int by = blockIdx.y;
         
    const int np = nh / C;
	const int tiles_x = np / blockDim.x + 1;
	const int tiles_y = np / blockDim.y + 1;

	for(int ix=0; ix < tiles_x; ix++) {
		for(int iy=0; iy < tiles_y; iy++){
			int idP_y = ty + iy * blockDim.y;
			int idP_x = tx + ix * blockDim.x;

			if(idP_y >= np || idP_x >= np) continue;

			float maxx = 0.0f;
			for(int k=0; k<C*C; k++){
				int idH_y = k / C + idP_y * C;
				int idH_x = k % C + idP_x * C;
				float h = d_H[nh*nh*gridDim.x*by + nh*nh*bx + nh*idH_y + idH_x];
				maxx = (h > maxx) ? h : maxx;
			}


			float summ = exp(0.0f-maxx);
			for(int k=0; k<C*C; k++){
				int idH_y = k / C + idP_y * C;
				int idH_x = k % C + idP_x * C;
				float h = d_H[nh*nh*gridDim.x*by + nh*nh*bx + nh*idH_y + idH_x];
				summ += exp(h-maxx);
			}		

			for(int k=0; k<C*C; k++){
				int idH_y = k / C + idP_y * C;
				int idH_x = k % C + idP_x * C;
				float h = d_H[nh*nh*gridDim.x*by + nh*nh*bx + nh*idH_y + idH_x];
				d_H[nh*nh*gridDim.x*by + nh*nh*bx + nh*idH_y + idH_x] = exp(h-maxx) / summ;
			}
		}
	}
}

__global__ void pooling_kernel(float *d_P, float *d_HS, int C, int nh)
{   
    const int tx = threadIdx.x; 
    const int ty = threadIdx.y;
    const int bx = blockIdx.x; 
    const int by = blockIdx.y;
         
    const int np = nh / C;
	const int tiles_x = np / blockDim.x + 1;
	const int tiles_y = np / blockDim.y + 1;

	for(int ix=0; ix < tiles_x; ix++) {
		for(int iy=0; iy < tiles_y; iy++){
			int idP_y = ty + iy * blockDim.y;
			int idP_x = tx + ix * blockDim.x;

			if(idP_y >= np || idP_x >= np) continue;

			float sum = 0.0f;
			for(int k=0; k<C*C; k++){
				int idH_y = k / C + idP_y * C;
				int idH_x = k % C + idP_x * C;
				sum += d_HS[nh*nh*gridDim.x*by + nh*nh*bx + nh*idH_y + idH_x];
			}			
			d_P[np*np*gridDim.x*by + np*np*bx + np*idP_y + idP_x] = sum / (C*C);
		}
	}
}

__global__ void prod3d_kernel(float *d_prod, float *d_V, float *d_H, int nv, int nh)
{   
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int bx = blockIdx.x; // kin id
	const int by = blockIdx.y; // kout id
	const int bz = blockIdx.z; // batch id

    const int nw = nv - nh + 1;
	const int tiles_x = nw / blockDim.x + 1;
	const int tiles_y = nw / blockDim.y + 1;

	for(int ix=0; ix < tiles_x; ix++){
		for(int iy=0; iy < tiles_y; iy++){
            int idW_y = ty + iy * blockDim.y; 
            int idW_x = tx + ix * blockDim.x;
            if(idW_y >= nw || idW_x >= nw) continue;

            float temp = 0.0f;
            for(int k=0; k<nh; k++){
                for(int l=0; l<nh; l++){
                    float v = d_V[nv*nv*gridDim.x*bz + nv*nv*bx + nv*(idW_y+k) + (idW_x+l)];
                    float h = d_H[nh*nh*gridDim.y*bz + nh*nh*by + nh*k + l];          
                    temp += v*h;
                }
            }
            atomicAdd(&d_prod[nw*nw*gridDim.x*by + nw*nw*bx + nw*idW_y + idW_x], temp);
        }
    }
}


__global__ void sum_along_channels_kernel(float *d_act, float *d_batches, int nv, int Kin, int batchsize)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x; // channel id
    const int by = blockIdx.y; // batch id

    const int tiles_x = nv / blockDim.x + 1;
    const int tiles_y = nv / blockDim.y + 1;

    for(int ix=0; ix < tiles_x; ix++){
        for(int iy=0; iy < tiles_y; iy++){
            int idx = tx + ix * blockDim.x;
            int idy = ty + iy * blockDim.y;
            if(idx >= nv || idy >= nv) continue;

            atomicAdd(&d_act[bx], d_batches[nv*nv*gridDim.x*by + nv*nv*bx + nv*ty + tx]);
        }
    }
}

__global__ void sum_along_channels_new_kernel(float *d_act, float * d_batches, int imsize, int channels, int batchsize)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < channels) {
		float temp = 0.0f;
		for(int i=0; i<imsize*imsize; i++){
			for(int j=0; j<batchsize; j++){
				temp += d_batches[imsize*imsize*channels*j + imsize*imsize*idx + i];
			}
		}
		d_act[idx] = temp;
	}
}

///////////////////// some utils //////////////////
__global__ void init_with_zero_kernel(float *d_A, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length){
        d_A[idx] = (float)0.0;
    }
}

__device__ float sigmoid(float x) {
	return (float) 1.0f / (1.0f + exp((-1.0f) * x));
}

__global__ void sigmoid_kernel(float *d_A, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length){
        d_A[idx] = sigmoid(d_A[idx]);
    }
}

__global__ void copy_kernel(float *d_des, float *d_src, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length){
        d_des[idx] = d_src[idx];
    }
}

__global__ void iadd_num_kernel(float *d_A, float x, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length){
        d_A[idx] += x;
    }
}

__global__ void isub_num_kernel(float *d_A, float x, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length){
        d_A[idx] -= x;
    }
}

__global__ void imul_num_kernel(float *d_A, float x, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length){
        d_A[idx] *= x;
    }
}

__global__ void idiv_num_kernel(float *d_A, float d, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length && fabs(d)>EPS){
        d_A[idx] /= d;
    }
}

__global__ void iadd_vec_kernel(float *d_A, float *d_B, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length){
        d_A[idx] += d_B[idx];
    }
}

__global__ void isub_vec_kernel(float *d_A, float *d_B, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length){
        d_A[idx] -= d_B[idx];
    }
}

__global__ void imul_vec_kernel(float *d_A, float *d_B, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length){
        d_A[idx] *= d_B[idx];
    }
}

__global__ void idiv_vec_kernel(float *d_A, float *d_B, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length && fabs(d_B[idx]) > EPS){    
        d_A[idx] /= d_B[idx];
    }
}

__global__ void add_kernel(float *d_ret, float *d_A, float *d_B, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length){
        d_ret[idx] = d_A[idx] + d_B[idx];
    }
}
    
__global__ void sub_kernel(float *d_ret, float *d_A, float *d_B, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length){
        d_ret[idx] = d_A[idx] - d_B[idx];
    }
}

__global__ void mul_kernel(float *d_ret, float *d_A, float *d_B, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length){
        d_ret[idx] = d_A[idx] * d_B[idx];
    }
}

__global__ void div_kernel(float *d_ret, float *d_A, float *d_B, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length && fabs(d_B[idx]) > EPS){
        d_ret[idx] = d_A[idx] / d_B[idx];
    }
}

__global__ void sub_num_kernel(float *d_ret, float *d_A, float x, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length){
        d_ret[idx] = d_A[idx] - x;
    }
}

__global__ void add_num_kernel(float *d_ret, float *d_A, float x, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length){
        d_ret[idx] = d_A[idx] + x;
    }
}

__global__ void mul_num_kernel(float *d_ret, float *d_A, float x, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length){
        d_ret[idx] = d_A[idx] * x;
    }
}

__global__ void div_num_kernel(float *d_ret, float *d_A, float x, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length){
        d_ret[idx] = d_A[idx] / x;
    }
}

__global__ void  truncate_kernel(float *d_A, float low, float high, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < length && d_A[idx] > high){
        d_A[idx] = high;
    }else if(idx < length && d_A[idx] < low){
        d_A[idx] = low;
    }
}
