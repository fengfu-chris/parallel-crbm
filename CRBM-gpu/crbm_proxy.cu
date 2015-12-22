#include <stdlib.h>
#include "./headers/utils.h"
#include "./headers/crbm.h"

void crbmTrain_proxy(float *W, float *vb, float *hb, float *patches, Parameters pars)
{
    float *d_W, *d_vb, *d_hb, *d_patches;

    size_t num_saves = pars.maxIter/pars.SAVE_PER_ITERS + 1;
    size_t size_W  = pars.nw * pars.nw * pars.Kin * pars.Kout;
    size_t size_vb = pars.Kin;
    size_t size_hb = pars.Kout;
    size_t size_patches = pars.nv * pars.nv * pars.Kin * pars.numsamples;

    //init_gaussian(W, 0, 0.1, size_W);
    //init_const(vb,  0.0, size_vb); 
    //init_const(hb, -0.1, size_hb);

    cudaMalloc((void**)&d_W,  size_W  * num_saves * sizeof(float));
    cudaMalloc((void**)&d_vb, size_vb * num_saves * sizeof(float));
    cudaMalloc((void**)&d_hb, size_hb * num_saves * sizeof(float));
    cudaMalloc((void**)&d_patches, size_patches * sizeof(float));

    cudaMemcpy(d_W,  W,  size_W  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hb, hb, size_hb * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vb, vb, size_vb * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_patches, patches, size_patches * sizeof(float), cudaMemcpyHostToDevice);

    crbmTrain(d_W, d_vb, d_hb, d_patches, pars);

    cudaMemcpy(W,  d_W,  size_W  * num_saves * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vb, d_vb, size_vb * num_saves * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb, d_hb, size_hb * num_saves * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_W);
    cudaFree(d_vb);
    cudaFree(d_hb);
    cudaFree(d_patches);
}

void crbmInfer_proxy(float *H, float *HS, float *V, float *W, float *hb, Parameters pars)
{
    float *d_H, *d_HS, *d_V, *d_W, *d_hb;
	
	size_t size_V_batch = pars.nv * pars.nv * pars.Kin  * pars.batchsize;
	size_t size_H_batch = pars.nh * pars.nh * pars.Kout * pars.batchsize;
    size_t size_W  = pars.nw * pars.nw * pars.Kin * pars.Kout;
    size_t size_hb = pars.Kout;

	cudaMalloc((void**)&d_H,  size_H_batch  * sizeof(float));
	cudaMalloc((void**)&d_HS, size_H_batch  * sizeof(float));
	cudaMalloc((void**)&d_V,  size_V_batch  * sizeof(float));
    cudaMalloc((void**)&d_W,  size_W  * sizeof(float));
    cudaMalloc((void**)&d_hb, size_hb * sizeof(float));

	cudaMemcpy(d_V,  V,  size_V_batch * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W,  W,  size_W  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hb, hb, size_hb * sizeof(float), cudaMemcpyHostToDevice);
   
	crbmInfer(d_H, d_HS, d_V, d_W, d_hb, pars);
    
    cudaMemcpy(H,  d_H,  size_H_batch * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(HS, d_HS, size_H_batch * sizeof(float), cudaMemcpyDeviceToHost);
   
    cudaFree(d_H);
    cudaFree(d_HS);
    cudaFree(d_V);
	cudaFree(d_W);
	cudaFree(d_hb);
}

void crbmRecon_proxy(float *V, float *H, float *W, float *vb, Parameters pars)
{
    float *d_V, *d_H, *d_W, *d_vb;
	
	size_t size_V_batch = pars.nv * pars.nv * pars.Kin  * pars.batchsize;
	size_t size_H_batch = pars.nh * pars.nh * pars.Kout * pars.batchsize;
    size_t size_W  = pars.nw * pars.nw * pars.Kin * pars.Kout;
    size_t size_vb = pars.Kin;

	cudaMalloc((void**)&d_V,  size_V_batch  * sizeof(float));
	cudaMalloc((void**)&d_H,  size_H_batch  * sizeof(float));
    cudaMalloc((void**)&d_W,  size_W  * sizeof(float));
    cudaMalloc((void**)&d_vb, size_vb * sizeof(float));

	cudaMemcpy(d_H,  H,  size_H_batch * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W,  W,  size_W  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vb, vb, size_vb * sizeof(float), cudaMemcpyHostToDevice);
   
	crbmRecon(d_V, d_H, d_W, d_vb, pars);
    
    cudaMemcpy(V,  d_V,  size_V_batch * sizeof(float), cudaMemcpyDeviceToHost);
   
    cudaFree(d_V);
    cudaFree(d_H);
	cudaFree(d_W);
	cudaFree(d_vb);
}

void crbmUpdates_proxy(float *dW, float *dvb, float *dhb, float *V, float *W, float *vb, float *hb, Parameters pars)
{
    float *d_V, *d_dW, *d_dvb, *d_dhb, *d_W, *d_vb, *d_hb;
	
	size_t size_V_batch = pars.nv * pars.nv * pars.Kin  * pars.batchsize;
    size_t size_W  = pars.nw * pars.nw * pars.Kin * pars.Kout;
    size_t size_vb = pars.Kin;
	size_t size_hb = pars.Kout;

	cudaMalloc((void**)&d_V,  size_V_batch  * sizeof(float));
    cudaMalloc((void**)&d_W,  size_W  * sizeof(float));
    cudaMalloc((void**)&d_vb, size_vb * sizeof(float));
	cudaMalloc((void**)&d_hb, size_hb * sizeof(float));
    cudaMalloc((void**)&d_dW,  size_W  * sizeof(float));
    cudaMalloc((void**)&d_dvb, size_vb * sizeof(float));
	cudaMalloc((void**)&d_dhb, size_hb * sizeof(float));

	cudaMemcpy(d_V,  V,  size_V_batch * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W,  W,  size_W  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vb, vb, size_vb * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_hb, hb, size_hb * sizeof(float), cudaMemcpyHostToDevice);   

	crbmUpdates(d_dW, d_dvb, d_dhb, d_V, d_W, d_vb, d_hb, pars);
    
    cudaMemcpy(dW,  d_dW,  size_W  * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(dvb, d_dvb, size_vb * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(dhb, d_dhb, size_hb * sizeof(float), cudaMemcpyDeviceToHost);
   
    cudaFree(d_V);
	cudaFree(d_W);
	cudaFree(d_vb);
	cudaFree(d_hb);
	cudaFree(d_dW);
	cudaFree(d_dvb);
	cudaFree(d_dhb);
}

void computeVisHid_proxy(float *prods, float *V, float *H, Parameters pars)
{
    float *d_prods, *d_V, *d_H;
	
	size_t size_V_batch = pars.nv * pars.nv * pars.Kin  * pars.batchsize;
    size_t size_H_batch = pars.nh * pars.nh * pars.Kout * pars.batchsize;
	size_t size_prods = pars.nw * pars.nw * pars.Kin * pars.Kout;

	cudaMalloc((void**)&d_V, size_V_batch * sizeof(float));
    cudaMalloc((void**)&d_H, size_H_batch * sizeof(float));
	cudaMalloc((void**)&d_prods, size_prods * sizeof(float));

	cudaMemcpy(d_V, V, size_V_batch * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_H, H, size_H_batch * sizeof(float), cudaMemcpyHostToDevice);   

	computeVisHid(d_prods, d_V, d_H, pars);
 
	cudaMemcpy(prods, d_prods, size_prods * sizeof(float), cudaMemcpyDeviceToHost);
   
    cudaFree(d_prods);
	cudaFree(d_V);
	cudaFree(d_H);
}

void computeAct_proxy(float *act, float *batches, int imsize, int channels, int batchsize)
{
    float *d_act, *d_batches;
	
	size_t size_batches = imsize * imsize * channels * batchsize;
    size_t size_act = channels;

	cudaMalloc((void**)&d_act, size_act * sizeof(float));
    cudaMalloc((void**)&d_batches, size_batches * sizeof(float));
	
	cudaMemcpy(d_batches, batches, size_batches * sizeof(float), cudaMemcpyHostToDevice);

	computeAct(d_act, d_batches, imsize, channels, batchsize);	

	cudaMemcpy(act, d_act, size_act * sizeof(float), cudaMemcpyDeviceToHost);
   
    cudaFree(d_act);
	cudaFree(d_batches);
}




