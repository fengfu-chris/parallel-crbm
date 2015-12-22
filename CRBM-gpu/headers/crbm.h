#ifndef _PARS_H_
#include "pars.h"
#endif

void crbmTrain(float *d_W, float *d_vb, float *d_hb, float *d_pathces, Parameters pars);
void crbmInfer(float *d_H, float *d_HS, float *d_V, float *d_W, float *d_hb, Parameters pars);
void crbmRecon(float *d_V, float *d_H, float *d_W, float *d_vb, Parameters pars);
void crbmUpdates(float *d_dW, float *d_dv, float *d_dh, float *d_V, float *d_W, float *d_vb, float *d_hb, Parameters pars);
void computeVisHid(float *d_prods, float *d_V, float *d_H, Parameters pars);
void computeAct(float *d_act, float *d_batches, int imsize, int channels, int batchsize);
