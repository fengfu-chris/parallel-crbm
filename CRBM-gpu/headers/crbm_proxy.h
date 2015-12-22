#ifndef _PARS_H_
#include "pars.h"
#endif

void print_pars(Parameters pars);

void crbmTrain_proxy(float *W, float *vb, float *hb, float *pathces, Parameters pars);
void crbmInfer_proxy(float *H, float *HS, float *V, float *W, float *hb, Parameters pars);
void crbmRecon_proxy(float *V, float *H, float *W, float *vb, Parameters pars);
void crbmUpdates_proxy(float *dW, float *dvb, float *dhb, float *V, float *W, float *vb, float *hb, Parameters pars);
void computeVisHid_proxy(float *prods, float *V, float *H, Parameters pars);
void computeAct_proxy(float *act, float *batches, int imsize, int channels, int batchsize);
