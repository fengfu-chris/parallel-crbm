#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include "./headers/kernels.h"
#include "./headers/kernels_proxy.h"

void sum_along_channels(float *act, float *batches, int nv, int Kin, int batchsize)
{
    float *d_act, *d_batches;

    size_t size_act = Kin;
    size_t size_batches = nv * nv * Kin * batchsize;

    cudaMalloc((void**)&d_act, size_act * sizeof(float));
    cudaMalloc((void**)&d_batches, size_batches * sizeof(float));

    cudaMemcpy(d_act, act, size_act * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_batches, batches, size_batches * sizeof(float), cudaMemcpyHostToDevice);

    init_with_zero_kernel<<< size_act / TPB_1d + 1, TPB_1d >>>(d_act, size_act);
    
    dim3 blockSize( TPB_2d, TPB_2d);
    dim3 gridSize( Kin, batchsize);
    sum_along_channels_kernel<<< gridSize, blockSize >>>(d_act, d_batches, nv, Kin, batchsize);
    
    cudaMemcpy(act, d_act, size_act * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceReset();
}


void filter3d_valid(float *H, float *V, float *W, int nv, int nw, int Kin, int Kout){
	float *d_V, *d_H, *d_W;

	const int nh = nv - nw + 1;

	cudaMalloc((void**)&d_H, nh * nh * Kout * sizeof(float));
	cudaMalloc((void**)&d_V, nv * nv * Kin  * sizeof(float));
	cudaMalloc((void**)&d_W, nw * nw * Kin * Kout * sizeof(float));

	cudaMemcpy(d_H, H, nh * nh * Kout * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, V, nv * nv * Kin  * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_W, W, nw * nw * Kin  * Kout * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockSize(TPB_2d, TPB_2d);
	dim3 gridSize(Kin, Kout);	

	filter3d_valid_kernel<<< gridSize, blockSize >>>(d_H, d_V, d_W, nv, nw);
	
	cudaMemcpy(H, d_H, nh * nh * Kout * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceReset();
}

void filter4d_valid(float *H, float *V, float *W, int nv, int nw, int Kin, int Kout, int batchsize){
	float *d_V, *d_H, *d_W;

	const int nh = nv - nw + 1;

	cudaMalloc((void**)&d_H, nh * nh * Kout * batchsize * sizeof(float));
	cudaMalloc((void**)&d_V, nv * nv * Kin  * batchsize * sizeof(float));
	cudaMalloc((void**)&d_W, nw * nw * Kin * Kout * sizeof(float));

	cudaMemcpy(d_H, H, nh * nh * Kout * batchsize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, V, nv * nv * Kin  * batchsize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_W, W, nw * nw * Kin * Kout * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockSize(TPB_2d, TPB_2d);
	dim3 gridSize(Kin, Kout, batchsize);	

	filter4d_valid_kernel<<< gridSize, blockSize >>>(d_H, d_V, d_W, nv, nw);
	
	cudaMemcpy(H, d_H, nh * nh * Kout * batchsize * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceReset();
}

void filter3d_full(float *V, float *H, float *W, int nh, int nw, int Kin, int Kout){
	float *d_V, *d_H, *d_W;

	const int nv = nh + nw - 1;

	cudaMalloc((void**)&d_H, nh * nh * Kout * sizeof(float));
	cudaMalloc((void**)&d_V, nv * nv * Kin  * sizeof(float));
	cudaMalloc((void**)&d_W, nw * nw * Kin * Kout * sizeof(float));

	cudaMemcpy(d_H, H, nh * nh * Kout * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, V, nv * nv * Kin  * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_W, W, nw * nw * Kin  * Kout * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockSize(TPB_2d, TPB_2d);
	dim3 gridSize(Kin, Kout);	

	filter3d_full_kernel<<< gridSize, blockSize >>>(d_V, d_H, d_W, nh, nw);
	
	cudaMemcpy(V, d_V, nv * nv * Kin * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceReset();
}

void filter4d_full(float *V, float *H, float *W, int nh, int nw, int Kin, int Kout, int batchsize){
	float *d_V, *d_H, *d_W;

	const int nv = nh + nw - 1;

	cudaMalloc((void**)&d_H, nh * nh * Kout * batchsize * sizeof(float));
	cudaMalloc((void**)&d_V, nv * nv * Kin * batchsize  * sizeof(float));
	cudaMalloc((void**)&d_W, nw * nw * Kin * Kout * sizeof(float));

	cudaMemcpy(d_H, H, nh * nh * Kout * batchsize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, V, nv * nv * Kin  * batchsize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_W, W, nw * nw * Kin  * Kout * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockSize(TPB_2d, TPB_2d);
	dim3 gridSize(Kin, Kout, batchsize);	

	filter4d_full_kernel<<< gridSize, blockSize >>>(d_V, d_H, d_W, nh, nw);
	
	cudaMemcpy(V, d_V, nv * nv * Kin * batchsize * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceReset();
}

void conv3d_valid(float *H, float *V, float *W, int nv, int nw, int Kin, int Kout){
	float *d_V, *d_H, *d_W;

	const int nh = nv - nw + 1;

	cudaMalloc((void**)&d_H, nh * nh * Kout * sizeof(float));
	cudaMalloc((void**)&d_V, nv * nv * Kin  * sizeof(float));
	cudaMalloc((void**)&d_W, nw * nw * Kin * Kout * sizeof(float));

	cudaMemcpy(d_H, H, nh * nh * Kout * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, V, nv * nv * Kin  * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_W, W, nw * nw * Kin  * Kout * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockSize(TPB_2d, TPB_2d);
	dim3 gridSize(Kin, Kout);	

	conv3d_valid_kernel<<< gridSize, blockSize >>>(d_H, d_V, d_W, nv, nw);
	
	cudaMemcpy(H, d_H, nh * nh * Kout * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceReset();
}

void conv4d_valid(float *H, float *V, float *W, int nv, int nw, int Kin, int Kout, int batchsize){
	float *d_V, *d_H, *d_W;

	const int nh = nv - nw + 1;

	cudaMalloc((void**)&d_H, nh * nh * Kout * batchsize * sizeof(float));
	cudaMalloc((void**)&d_V, nv * nv * Kin  * batchsize * sizeof(float));
	cudaMalloc((void**)&d_W, nw * nw * Kin * Kout * sizeof(float));

	cudaMemcpy(d_H, H, nh * nh * Kout * batchsize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, V, nv * nv * Kin  * batchsize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_W, W, nw * nw * Kin * Kout * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockSize(TPB_2d, TPB_2d);
	dim3 gridSize(Kin, Kout, batchsize);	

	conv4d_valid_kernel<<< gridSize, blockSize >>>(d_H, d_V, d_W, nv, nw);
	
	cudaMemcpy(H, d_H, nh * nh * Kout * batchsize * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceReset();
}

void conv3d_full(float *V, float *H, float *W, int nh, int nw, int Kin, int Kout){
	float *d_V, *d_H, *d_W;

	const int nv = nh + nw - 1;

	cudaMalloc((void**)&d_H, nh * nh * Kout * sizeof(float));
	cudaMalloc((void**)&d_V, nv * nv * Kin  * sizeof(float));
	cudaMalloc((void**)&d_W, nw * nw * Kin * Kout * sizeof(float));

	cudaMemcpy(d_H, H, nh * nh * Kout * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, V, nv * nv * Kin  * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_W, W, nw * nw * Kin  * Kout * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockSize(TPB_2d, TPB_2d);
	dim3 gridSize(Kin, Kout);	

	conv3d_full_kernel<<< gridSize, blockSize >>>(d_V, d_H, d_W, nh, nw);
	
	cudaMemcpy(V, d_V, nv * nv * Kin * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceReset();
}

void conv4d_full(float *V, float *H, float *W, int nh, int nw, int Kin, int Kout, int batchsize){
	float *d_V, *d_H, *d_W;

	const int nv = nh + nw - 1;

	cudaMalloc((void**)&d_H, nh * nh * Kout * batchsize * sizeof(float));
	cudaMalloc((void**)&d_V, nv * nv * Kin * batchsize  * sizeof(float));
	cudaMalloc((void**)&d_W, nw * nw * Kin * Kout * sizeof(float));

	cudaMemcpy(d_H, H, nh * nh * Kout * batchsize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, V, nv * nv * Kin  * batchsize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_W, W, nw * nw * Kin  * Kout * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockSize(TPB_2d, TPB_2d);
	dim3 gridSize(Kin, Kout, batchsize);	

	conv4d_full_kernel<<< gridSize, blockSize >>>(d_V, d_H, d_W, nh, nw);
	
	cudaMemcpy(V, d_V, nv * nv * Kin * batchsize * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceReset();
}

void mapping(float *H, float *hb, int nh, int Kout, int batchsize) {
	float *d_H, *d_hb;

	cudaMalloc((void**)&d_H, nh * nh * Kout * batchsize * sizeof(float));
	cudaMalloc((void**)&d_hb, Kout * sizeof(float));

	cudaMemcpy(d_H, H, nh * nh * Kout * batchsize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_hb, hb, Kout * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockSize(TPB_2d, TPB_2d);
	dim3 gridSize(Kout, batchsize);

	mapping_kernel<<< gridSize, blockSize >>>(d_H, d_hb, nh);

	cudaMemcpy(H, d_H, nh * nh * Kout * batchsize * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceReset();
}

void pooling(float *P, float *HS, int C, int nh, int Kout, int batchsize){
	float *d_P, *d_HS;

	const int np = nh / C;

	cudaMalloc((void**)&d_P,  np * np * Kout * batchsize * sizeof(float));
	cudaMalloc((void**)&d_HS, nh * nh * Kout * batchsize * sizeof(float));

	cudaMemcpy(d_P, P,   np * np * Kout * batchsize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_HS, HS, nh * nh * Kout * batchsize * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockSize(TPB_2d, TPB_2d);
	dim3 gridSize(Kout, batchsize);

	pooling_kernel<<< gridSize, blockSize >>>(d_P, d_HS, C, nh);

	cudaMemcpy(P, d_P, np * np * Kout * batchsize * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceReset();
}

void prod3d(float *prod, float *V, float *H, int nv, int nh, int Kin, int Kout, int batchsize) {
	float *d_prod, *d_V, *d_H;

	const int nw = nv - nh + 1;

	int size_of_prod = nw * nw * Kin * Kout;
	int size_of_V = nv * nv * Kin * batchsize;
	int size_of_H = nh * nh * Kout * batchsize;

	cudaMalloc((void**)&d_prod, size_of_prod * sizeof(float));
	cudaMalloc((void**)&d_V,  size_of_V * sizeof(float));
	cudaMalloc((void**)&d_H,  size_of_H * sizeof(float));

	cudaMemcpy(d_prod, prod,   size_of_prod * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, V, size_of_V * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_H, H, size_of_H * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockSize(TPB_2d, TPB_2d);
	dim3 gridSize(Kin, Kout, batchsize);

	prod3d_kernel<<< gridSize, blockSize >>>(d_prod, d_V, d_H, nv, nh);

	cudaMemcpy(prod, d_prod, size_of_prod * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceReset();
}

