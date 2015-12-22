float gaussrand(void);

void init_gaussian(float *A, float mean, float std, int length);
void init_const(float *A, float c, int length);
void init_rand(float *A, long seed, int length);
 
void print1d(float *A, int length);
void print2d(float *A, int numRows, int numCols);

void copy(float *des, float *res, int length);

float compute_mean(float *A, int length);
float compute_stdvar(float *A, int length);

void add_num(float *A, float num, int length);
void sub_num(float *A, float num, int length);
void mul_num(float *A, float scaler, int length);
void div_num(float *A, float d, int length);
void add_vec(float *A, float *B, int length);
void sub_vec(float *A, float *B, int length);
void mul_vec(float *A, float *B, int length);
void div_vec(float *A, float *B, int length);
void add(float *result, float *A, float *B, int length);
void sub(float *result, float *A, float *B, int length);
void mul(float *results, float *A, float *B, int length);
void div(float *results, float *A, float *B, int length);

// Dealing with device data
void print2d_device(float *d_A, int numRows, int numCols);
void init_rand_device(float *d_r, long seed, int length);
