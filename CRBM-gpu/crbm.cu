#include <mex.h>
#include <stdlib.h>
#include "./headers/defines.h"
#include "./headers/utils.h"
#include "./headers/kernels.h"
#include "./headers/crbm.h"

void print_pars(Parameters pars)
{
    mexPrintf("\n===== printing the parameters =====\n\n");
    mexPrintf("nv        = %d\n", pars.nv);
    mexPrintf("nh        = %d\n", pars.nh);
    mexPrintf("nw        = %d\n", pars.nw);
    mexPrintf("C         = %d\n", pars.C);
    mexPrintf("np        = %d\n", pars.np);
    mexPrintf("Kin       = %d\n", pars.Kin);
    mexPrintf("Kout      = %d\n", pars.Kout);
    mexPrintf("epsilon   = %g\n", pars.epsilon);
    mexPrintf("momemtum  = %g\n", pars.momemtum);
    mexPrintf("l2reg     = %g\n", pars.l2reg);
    mexPrintf("pbias     = %g\n", pars.pbias);
    mexPrintf("pbiasL    = %g\n", pars.pbiasL);
    mexPrintf("std_gau.  = %g\n", pars.std_gaussian);
    mexPrintf("maxIter   = %d\n", pars.maxIter);
    mexPrintf("SAVE_ITER = %d\n", pars.SAVE_PER_ITERS);
    mexPrintf("batchsize = %d\n", pars.batchsize);
    mexPrintf("numsamp.  = %d\n", pars.numsamples);
    mexPrintf("Vtype     = %s\n", pars.Vtype);
    mexPrintf("Htype     = %s\n", pars.Htype);
    mexPrintf("DEBUG     = %s\n", pars.DEBUG);
    mexPrintf("\n===== end of the printing =====\n");
}

void crbmTrain(float *d_W, float *d_vb, float *d_hb, float *d_patches, Parameters pars)
{     
    float *d_dW, *d_dhb, *d_dvb, *d_Winc, *d_hbinc, *d_vbinc;

    size_t size_V  = pars.nv * pars.nv * pars.Kin;
    size_t size_W  = pars.nw * pars.nw * pars.Kin  * pars.Kout;
    size_t size_vb = pars.Kin;
    size_t size_hb = pars.Kout;

    cudaMalloc((void**)&d_dW,    size_W  * sizeof(float));
    cudaMalloc((void**)&d_dvb,   size_vb * sizeof(float));
    cudaMalloc((void**)&d_dhb,   size_hb * sizeof(float));
    cudaMalloc((void**)&d_Winc,  size_W  * sizeof(float));    
    cudaMalloc((void**)&d_vbinc, size_vb * sizeof(float));	
    cudaMalloc((void**)&d_hbinc, size_hb * sizeof(float));
	
    init_with_zero_kernel<<< size_W /TPB_1d + 1, TPB_1d >>>(d_Winc,  size_W);
    init_with_zero_kernel<<< size_vb/TPB_1d + 1, TPB_1d >>>(d_vbinc, size_vb);
    init_with_zero_kernel<<< size_hb/TPB_1d + 1, TPB_1d >>>(d_hbinc, size_hb);

    float *d_W_curr  = d_W;
    float *d_vb_curr = d_vb;
    float *d_hb_curr = d_hb;

    float *d_W_prev; 
    float *d_vb_prev; 
    float *d_hb_prev;

    for(int it = 0; it < pars.maxIter; it++)
    {
        if(strcmp(pars.DEBUG, "yes") == 0 && (it==0 || (it+1) % pars.SAVE_PER_ITERS == 0)){
			print2d_device(d_W_curr,  1, pars.nw * pars.nw);
			// print2d_device(d_hb_curr, 1, size_hb);
			// print2d_device(d_vb_curr, 1, size_vb);
        }

		if(it % 1000 == 0){
			// mexPrintf("it = %d\n", it);
		}

        if( (it+1) % pars.SAVE_PER_ITERS == 0) {
            d_W_prev  = d_W_curr;
            d_vb_prev = d_vb_curr;
            d_hb_prev = d_hb_curr;

            d_W_curr  = d_W_prev  + size_W;
            d_vb_curr = d_vb_prev + size_vb;
            d_hb_curr = d_hb_prev + size_hb;
    
        	copy_kernel<<< size_W /TPB_1d + 1, TPB_1d >>>(d_W_curr,  d_W_prev,  size_W);
            copy_kernel<<< size_vb/TPB_1d + 1, TPB_1d >>>(d_vb_curr, d_vb_prev, size_vb);	
            copy_kernel<<< size_hb/TPB_1d + 1, TPB_1d >>>(d_hb_curr, d_hb_prev, size_hb);
        }

        pars.momemtum = (it < 20000) ? 0.5 : 0.9;

        // int batchStartId = rand() * rand() % (pars.numsamples - pars.batchsize + 1);
	int batchStartId = (it * pars.batchsize) % (pars.numsamples - pars.batchsize + 1);
        float *d_V = d_patches + size_V * batchStartId;      

        crbmUpdates(d_dW, d_dvb, d_dhb, d_V, d_W_curr, d_vb_curr, d_hb_curr, pars);

        // compute Winc
        imul_num_kernel<<< size_W/TPB_1d + 1, TPB_1d >>>(d_Winc, pars.momemtum, size_W);
        imul_num_kernel<<< size_W/TPB_1d + 1, TPB_1d >>>(d_dW, pars.epsilon, size_W);
        iadd_vec_kernel<<< size_W/TPB_1d + 1, TPB_1d >>>(d_Winc, d_dW, size_W);

        // compute vbinc
        imul_num_kernel<<< size_vb/TPB_1d + 1, TPB_1d >>>(d_vbinc, pars.momemtum, size_vb);
        imul_num_kernel<<< size_vb/TPB_1d + 1, TPB_1d >>>(d_dvb, pars.epsilon, size_vb);        
        iadd_vec_kernel<<< size_vb/TPB_1d + 1, TPB_1d >>>(d_vbinc, d_dvb, size_vb);

        // compute hbinc
        imul_num_kernel<<< size_hb/TPB_1d + 1, TPB_1d >>>(d_hbinc, pars.momemtum, size_hb);
        imul_num_kernel<<< size_hb/TPB_1d + 1, TPB_1d >>>(d_dhb, pars.epsilon, size_hb);
        iadd_vec_kernel<<< size_hb/TPB_1d + 1, TPB_1d >>>(d_hbinc, d_dhb, size_hb); 

        // update W, vb and hb
        iadd_vec_kernel<<< size_W /TPB_1d + 1, TPB_1d >>>(d_W_curr,  d_Winc,  size_W);
        iadd_vec_kernel<<< size_vb/TPB_1d + 1, TPB_1d >>>(d_vb_curr, d_vbinc, size_vb);
        iadd_vec_kernel<<< size_hb/TPB_1d + 1, TPB_1d >>>(d_hb_curr, d_hbinc, size_hb);
    }

    cudaFree(d_dW);
    cudaFree(d_dvb);
    cudaFree(d_dhb);
    cudaFree(d_Winc);
    cudaFree(d_vbinc);
    cudaFree(d_hbinc);
}

void crbmInfer(float *d_H, float *d_HS, float *d_V, float *d_W, float *d_hb, Parameters pars)
{
    size_t size_H_batch = pars.nh * pars.nh * pars.Kout * pars.batchsize;
    size_t size_P_batch = pars.np * pars.np * pars.Kout * pars.batchsize;
    
    init_with_zero_kernel<<<size_H_batch/TPB_1d + 1, TPB_1d>>>(d_H,  size_H_batch);
    init_with_zero_kernel<<<size_H_batch/TPB_1d + 1, TPB_1d>>>(d_HS, size_H_batch);

    float *d_r;       
    cudaMalloc((void**)&d_r, size_P_batch * sizeof(float));
    long seed = rand()*rand() % 1000000;
    // init_rand_device(d_r, seed, size_P_batch);
  
    dim3 blockSize(TPB_2d, TPB_2d);
    dim3 gridSize(pars.Kin, pars.Kout, pars.batchsize);        
    filter4d_valid_kernel<<< gridSize, blockSize >>>(d_H, d_V, d_W, pars.nv, pars.nw);
    
    dim3 gridSize2(pars.Kout, pars.batchsize);
    mapping_kernel<<< gridSize2, blockSize >>>(d_H, d_hb, pars.nh);
    
    idiv_num_kernel<<< size_H_batch/TPB_1d + 1, TPB_1d >>>(d_H, pars.std_gaussian, size_H_batch);
    
    // fix this!!!
    // truncate_kernel<<< size_H_batch/TPB_1d + 1, TPB_1d >>>(d_H, -10.0, 10.0, size_H_batch);
    
    dim3 gridSize3(pars.Kout, pars.batchsize);
    sampling_kernel<<< gridSize3, blockSize >>>(d_H, d_HS, d_r, pars.C, pars.nh);
    // end of the fix
    
    cudaFree(d_r);
}

void crbmRecon(float *d_V, float *d_H, float *d_W, float *d_vb, Parameters pars)
{
    size_t size_V_batch = pars.nv * pars.nv * pars.Kin * pars.batchsize;
	init_with_zero_kernel<<< size_V_batch/TPB_1d + 1, TPB_1d >>>(d_V, size_V_batch);

	dim3 blockSize( TPB_2d, TPB_2d );
	dim3 gridSize( pars.Kin, pars.Kout, pars.batchsize);

	conv4d_full_kernel<<< gridSize, blockSize >>>(d_V, d_H, d_W, pars.nh, pars.nw);
        
	dim3 gridSize2(pars.Kin, pars.batchsize);
	mapping_kernel<<< gridSize2, blockSize >>>(d_V, d_vb, pars.nv);

    if(strcmp(pars.Vtype, "binary") == 0){
		truncate_kernel<<< size_V_batch/TPB_1d + 1, TPB_1d >>>(d_V, -80.0, 80.0, size_V_batch);
		sigmoid_kernel<<< size_V_batch/TPB_1d + 1, TPB_1d >>>(d_V, size_V_batch);
    }     
}

void crbmUpdates(float *d_dW, float *d_dvb, float *d_dhb, 
    float *d_V, float *d_W, float *d_vb, float *d_hb, Parameters pars)
{

    float *d_H, *d_HS, *d_Vneg, *d_Hneg, *d_HSneg;
    float *d_posprods, *d_negprods, *d_posvisact, *d_negvisact, *d_poshidact, *d_neghidact;
    float *d_dW2, *d_db_sp;

    size_t size_V_batch = pars.nv * pars.nv * pars.Kin  * pars.batchsize;
    size_t size_H_batch = pars.nh * pars.nh * pars.Kout * pars.batchsize;
    size_t size_W = pars.nw * pars.nw * pars.Kin * pars.Kout;
    size_t size_vb = pars.Kin;
    size_t size_hb = pars.Kout;

    cudaMalloc((void**)&d_H,         size_H_batch  * sizeof(float));
    cudaMalloc((void**)&d_HS,        size_H_batch  * sizeof(float));
    cudaMalloc((void**)&d_Vneg,      size_V_batch  * sizeof(float));
    cudaMalloc((void**)&d_Hneg,      size_H_batch  * sizeof(float));
    cudaMalloc((void**)&d_HSneg,     size_H_batch  * sizeof(float));
    cudaMalloc((void**)&d_posprods,  size_W  * sizeof(float));
    cudaMalloc((void**)&d_negprods,  size_W  * sizeof(float));
    cudaMalloc((void**)&d_posvisact, size_vb * sizeof(float));
    cudaMalloc((void**)&d_negvisact, size_vb * sizeof(float));    
    cudaMalloc((void**)&d_poshidact, size_hb * sizeof(float));
    cudaMalloc((void**)&d_neghidact, size_hb * sizeof(float));
    cudaMalloc((void**)&d_dW2,       size_W  * sizeof(float));
    cudaMalloc((void**)&d_db_sp,     size_hb * sizeof(float));

    init_with_zero_kernel<<< size_W  / TPB_1d + 1, TPB_1d >>>(d_dW,  size_W);
    init_with_zero_kernel<<< size_vb / TPB_1d + 1, TPB_1d >>>(d_dvb, size_vb);
    init_with_zero_kernel<<< size_hb / TPB_1d + 1, TPB_1d >>>(d_dhb, size_hb);

    init_with_zero_kernel<<< size_W  / TPB_1d + 1, TPB_1d >>>(d_dW2, size_W);
    init_with_zero_kernel<<< size_hb / TPB_1d + 1, TPB_1d >>>(d_db_sp, size_hb);

    // POSITIVE AND NEGATIVE PHASE
    crbmInfer(d_H, d_HS, d_V, d_W, d_hb, pars);
    crbmRecon(d_Vneg, d_H, d_W, d_vb, pars);
    crbmInfer(d_Hneg, d_HSneg, d_Vneg, d_W, d_hb, pars);

	// Compute posprods, negprods, posvisact, negvisact, poshidact, and neghidact
    computeVisHid(d_posprods, d_V,    d_H,    pars);
    computeVisHid(d_negprods, d_Vneg, d_Hneg, pars);

    computeAct(d_posvisact, d_V,    pars.nv, pars.Kin, pars.batchsize);
    computeAct(d_negvisact, d_Vneg, pars.nv, pars.Kin, pars.batchsize);
    
    computeAct(d_poshidact, d_H,    pars.nh, pars.Kout, pars.batchsize);
    computeAct(d_neghidact, d_Hneg, pars.nh, pars.Kout, pars.batchsize);

    // UPDATE WEIGHTS AND BIASES 
    sub_kernel<<< size_W / TPB_1d + 1, TPB_1d >>>(d_dW, d_posprods, d_negprods, size_W);
    idiv_num_kernel<<< size_W / TPB_1d + 1, TPB_1d >>>(d_dW, (float)(pars.nh * pars.nh), size_W);
    mul_num_kernel<<< size_W / TPB_1d + 1, TPB_1d >>>(d_dW2, d_W, (-1.0f)*pars.l2reg, size_W);
    iadd_vec_kernel<<< size_W / TPB_1d + 1, TPB_1d >>>(d_dW, d_dW2, size_W);

    sub_kernel<<< size_vb / TPB_1d + 1, TPB_1d >>>(d_dvb, d_posvisact, d_negvisact, size_vb);    

    sub_kernel<<< size_hb / TPB_1d + 1, TPB_1d >>>(d_dhb, d_poshidact, d_neghidact, size_hb);
    sub_num_kernel<<< size_hb / TPB_1d + 1, TPB_1d >>>(d_db_sp, d_poshidact, pars.pbias, size_hb);
    imul_num_kernel<<< size_hb / TPB_1d + 1, TPB_1d >>>(d_db_sp, (-1.0f)*pars.pbiasL, size_hb);
    iadd_vec_kernel<<< size_hb / TPB_1d + 1, TPB_1d >>>(d_dhb, d_db_sp, size_hb);

    cudaFree(d_H);
    cudaFree(d_HS);
    cudaFree(d_Vneg);
    cudaFree(d_Hneg);
    cudaFree(d_HSneg);
    cudaFree(d_posprods);
    cudaFree(d_negprods);
    cudaFree(d_posvisact);
    cudaFree(d_poshidact);
    cudaFree(d_negvisact);
    cudaFree(d_neghidact);
    cudaFree(d_dW2);
    cudaFree(d_db_sp);
}

void computeVisHid(float *d_prods, float *d_V, float *d_H, Parameters pars)
{
    size_t size_W  = pars.nw * pars.nw * pars.Kin * pars.Kout;
	init_with_zero_kernel<<< size_W /TPB_1d + 1, TPB_1d >>>(d_prods, size_W);

	dim3 blockSize( TPB_2d, TPB_2d );
	dim3 gridSize( pars.Kin, pars.Kout, pars.batchsize );
	prod3d_kernel<<< gridSize, blockSize >>>(d_prods, d_V, d_H, pars.nv, pars.nh);
	idiv_num_kernel<<< size_W/TPB_1d + 1, TPB_1d >>>(d_prods, pars.batchsize, size_W);
}

void computeAct(float *d_act, float *d_batches, int imsize, int channels, int batchsize){
    init_with_zero_kernel<<< channels / TPB_1d + 1, TPB_1d >>>(d_act, channels);

    //dim3 blockSize( TPB_2d, TPB_2d);
    //dim3 gridSize( channels, batchsize);
    //sum_along_channels_kernel<<< gridSize, blockSize >>>(d_act, d_batches, imsize, channels, batchsize);

	sum_along_channels_new_kernel<<< channels / TPB_1d + 1, TPB_1d >>>(d_act, d_batches, imsize, channels, batchsize);

    idiv_num_kernel<<< channels / TPB_1d + 1, TPB_1d >>>(d_act, (float) imsize * imsize * batchsize, channels);
}
