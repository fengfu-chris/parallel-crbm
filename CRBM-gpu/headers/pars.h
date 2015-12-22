#ifndef _PARS_H_
#define _PARS_H_
struct Parameters
{    
    int nv;
	int nh;
	int nw;
    int C;
	int np;
    int Kin;
	int Kout;
    float epsilon;
    float momemtum;
	float l2reg;
	float pbias;
	float pbiasL;
    float std_gaussian;
    int maxIter;
    int SAVE_PER_ITERS;
	int batchsize;
	int numsamples;
    char *Vtype;
    char *Htype;
    char *DEBUG;
};
#endif
