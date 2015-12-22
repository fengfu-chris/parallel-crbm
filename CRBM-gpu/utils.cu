#include <mex.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include "./headers/utils.h"
#include "./headers/defines.h"

float gaussrand(void)
{
     float r = 0.0f;
     for(int i = 0; i < NSUM; i++)
     {
         r += (float)rand() / RAND_MAX;
     }

     r -= NSUM / 2.0;
     r /= sqrt(NSUM / 12.0);

     return r;
}

void init_gaussian(float *A, float mean, float std, int length)
{
    for(int i=0; i<length; i++){
        A[i] = mean + std * gaussrand();
    }
}

void init_const(float *A, float c, int length)
{
    for(int i=0; i<length; i++){
        A[i] = c;
    }
}

void init_rand(float *A, long seed, int length) {
    for(int i=0; i<length; i++){
        A[i] = (float)rand() / RAND_MAX;
    }
}

void print1d(float *A, int length) {
	for(int i=0; i<length; i++) {
		mexPrintf("%5.2f ", A[i]);
	}
}

void print2d(float *A, int numRows, int numCols)
{ 
    for(int i=0; i<numRows; i++){
        for(int j=0; j<numCols; j++){
            mexPrintf("%5.2f ",A[i*numCols+j]);
        }
        mexPrintf("\n");
    }
    mexPrintf("\n");
}

void copy(float *des, float *res, int length)
{
    for(int i=0; i<length; i++){
        des[i] = res[i];
    }
}

float compute_mean(float *A, int length)
{
    float sum = 0.0;
    for(int i=0; i<length; i++){
        sum += A[i];
    }
    
    return sum/length;
}

float compute_stdvar(float *A, int length)
{
    float mean = compute_mean(A, length);
    float variance = 0.0;
    for(int i=0; i<length; i++){
        variance += pow(A[i]-mean, 2);
    }
    variance /= length - 1;
    
    return sqrt(variance);
}


void add_num(float *A, float num, int length)
{
    for(int i=0; i<length; i++){
        A[i] += num;
    }
}

void sub_num(float *A, float num, int length)
{
    for(int i=0; i<length; i++){
        A[i] -= num;
    }
}

void mul_num(float *A, float scaler, int length)
{
    for(int i=0; i<length; i++){
        A[i] *= scaler;
    }
}

void div_num(float *A, float d, int length)
{
    if(fabs(d) < EPS){
        mexPrintf("Error! Can not divide by 0!\n");
        return;
    }
    for(int i=0; i<length; i++){
        A[i] /= d;
    }
}

void add_vec(float *A, float *B, int length)
{
    for(int i=0; i<length; i++){
        A[i] += B[i];
    }
}

void sub_vec(float *A, float *B, int length)
{
    for(int i=0; i<length; i++){
        A[i] -= B[i];
    }
}

void mul_vec(float *A, float *B, int length)
{
    for(int i=0; i<length; i++){
        A[i] *= B[i];
    }
}

void div_vec(float *A, float *B, int length)
{
    for(int i=0; i<length; i++){
        if(B[i] < EPS){
            mexPrintf("Error! Can not divide by 0!\n");
            return;
        }
        A[i] /= B[i];
    }
}

void add(float *result, float *A, float *B, int length)
{
    for(int i=0; i<length; i++){
        result[i] = A[i] + B[i];
    }
}

void sub(float *result, float *A, float *B, int length)
{
    for(int i=0; i<length; i++){
        result[i] = A[i] - B[i];
    }
}

void mul(float *results, float *A, float *B, int length)
{
    for(int i=0; i<length; i++){
        results[i] = A[i] * B[i];
    }
}

void div(float *results, float *A, float *B, int length)
{
    for(int i=0; i<length; i++){
        if(fabs(B[i]) < EPS){
            mexPrintf("Error! Can not divide by 0!\n");
            return;
        }
        results[i] = A[i] / B[i];
    }
}

// Dealing with device data
void print2d_device(float *d_A, int numRows, int numCols)
{
    size_t size_A = numRows * numCols;
    float *A = (float*)malloc(size_A * sizeof(float));
    cudaMemcpy(A, d_A, size_A * sizeof(float), cudaMemcpyDeviceToHost);
    
	print2d(A, numRows, numCols);

    free(A);
}

void init_rand_device(float *d_r, long seed, int length){
    curandGenerator_t gen;  
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    
    curandGenerateUniform(gen, d_r, length);
 
    curandDestroyGenerator(gen);
}

