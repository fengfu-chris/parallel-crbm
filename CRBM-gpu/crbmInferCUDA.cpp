#include "mex.h"
#include <math.h>
#include "./headers/utils.h"
#include "./headers/crbm.h"
#include "./headers/crbm_proxy.h"

// matlab API: [H, HS] = crbmInferCUDA(V, W, hb, pars)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{  
    float *V = (float*)mxGetData(prhs[0]);
    float *W = (float*)mxGetData(prhs[1]);
    float *hb = (float*)mxGetData(prhs[2]);
    const mxArray *p_pars = prhs[3];
    
    int *p_nv = (int*)mxGetData(mxGetField(p_pars, 0, "nv"));
    int *p_nw = (int*)mxGetData(mxGetField(p_pars, 0, "nw"));
    int *p_C = (int*)mxGetData(mxGetField(p_pars, 0, "C"));
    int *p_Kin = (int*)mxGetData(mxGetField(p_pars, 0, "Kin"));
    int *p_Kout = (int*)mxGetData(mxGetField(p_pars, 0, "Kout"));
    float *p_epsilon = (float*)mxGetData(mxGetField(p_pars, 0, "epsilon"));
    float *p_momemtum = (float*)mxGetData(mxGetField(p_pars, 0, "momemtum"));
    float *p_l2reg = (float*)mxGetData(mxGetField(p_pars, 0, "l2reg"));
    float *p_pbias = (float*)mxGetData(mxGetField(p_pars, 0, "pbias"));
    float *p_pbiasL = (float*)mxGetData(mxGetField(p_pars, 0, "pbiasL"));
    float *p_std_gaussian = (float*)mxGetData(mxGetField(p_pars, 0, "std_gaussian"));
    int *p_maxIter = (int*)mxGetData(mxGetField(p_pars, 0, "maxIter"));
    int *p_SAVE_PER_ITERS = (int*)mxGetData(mxGetField(p_pars, 0, "SAVE_PER_ITERS"));
    int *p_batchsize = (int*)mxGetData(mxGetField(p_pars, 0, "batchsize"));
    int *p_numsamples = (int*)mxGetData(mxGetField(p_pars, 0, "numsamples"));
    char *p_Vtype = mxArrayToString(mxGetField(p_pars, 0, "Vtype"));
    char *p_Htype = mxArrayToString(mxGetField(p_pars, 0, "Htype"));
    char *p_DEBUG = mxArrayToString(mxGetField(p_pars, 0, "DEBUG"));
    
    Parameters pars;
    pars.nv = *p_nv;
    pars.nh = *p_nv - *p_nw + 1;
    pars.nw = *p_nw;
    pars.C = *p_C;
    pars.np = pars.nh / pars.C;
    pars.Kin = *p_Kin;
    pars.Kout = *p_Kout;
    pars.epsilon = *p_epsilon;
    pars.momemtum = *p_momemtum;
    pars.l2reg = *p_l2reg;
    pars.pbias = *p_pbias;
    pars.pbiasL = *p_pbiasL;
    pars.std_gaussian = *p_std_gaussian;
    pars.maxIter = *p_maxIter;
    pars.SAVE_PER_ITERS = *p_SAVE_PER_ITERS;
    pars.batchsize = *p_batchsize;
    pars.numsamples = *p_numsamples;
    pars.Vtype = p_Vtype;
    pars.Htype = p_Htype;
    pars.DEBUG = p_DEBUG;
    
    print_pars(pars);
    
    size_t size_H_batch = pars.nh * pars.nh * pars.Kout * pars.batchsize;
    
    plhs[0] = mxCreateNumericMatrix(size_H_batch, 1, mxSINGLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(size_H_batch, 1, mxSINGLE_CLASS, mxREAL);
    float *H  = (float*)mxGetData(plhs[0]);
    float *HS = (float*)mxGetData(plhs[1]);

	init_const(H,  0, size_H_batch);
	init_const(HS, 0, size_H_batch);

	crbmInfer_proxy(H, HS, V, W, hb, pars);
}
