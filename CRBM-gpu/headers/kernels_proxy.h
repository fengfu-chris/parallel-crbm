void conv3d_valid(float *H, float *V, float *W, int nv, int nw, int Kin, int Kout);

void conv4d_valid(float *H, float *V, float *W, int nv, int nw, int Kin, int Kout, int batchsize);

void conv3d_full(float *V, float *H, float *W, int nh, int nw, int Kin, int Kout);

void conv4d_full(float *V, float *H, float *W, int nh, int nw, int Kin, int Kout, int batchsize);

void filter3d_valid(float *H, float *V, float *W, int nv, int nw, int Kin, int Kout);

void filter4d_valid(float *H, float *V, float *W, int nv, int nw, int Kin, int Kout, int batchsize);

void filter3d_full(float *V, float *H, float *W, int nh, int nw, int Kin, int Kout);

void filter4d_full(float *V, float *H, float *W, int nh, int nw, int Kin, int Kout, int batchsize);

void mapping(float *H, float *hb, int nh, int Kout, int batchsize);

void pooling(float *P, float *HS, int C, int nh, int Kout, int batchsize);

void prod3d(float *prod, float *V, float *H, int nv, int nh, int Kin, int Kout, int batchsize);

