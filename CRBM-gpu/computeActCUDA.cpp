#include "mex.h"
#include <math.h>
#include "./headers/utils.h"
#include "./headers/crbm.h"
#include "./headers/crbm_proxy.h"

// matlab API: [act] = computeActCUDA(batches, imsize, channels, batchsize);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{  
    float *batches = (float*)mxGetData(prhs[0]);
	int *pimsize = (int*)mxGetData(prhs[1]);
	int *pchannels = (int*)mxGetData(prhs[2]);
	int *pbatchsize = (int*)mxGetData(prhs[3]);

    size_t size_act = *pchannels;
    
    plhs[0] = mxCreateNumericMatrix(size_act, 1, mxSINGLE_CLASS, mxREAL);
    float *act  = (float*)mxGetData(plhs[0]);
  
	init_const(act, 0, size_act);
	
	computeAct_proxy(act, batches, *pimsize, *pchannels, *pbatchsize);
}
