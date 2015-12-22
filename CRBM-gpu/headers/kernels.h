__global__ void conv3d_valid_kernel(float *d_H, float *d_V, float *d_W, int nv, int nw);
__global__ void conv4d_valid_kernel(float *d_H, float *d_V, float *d_W, int nv, int nw);
__global__ void conv3d_full_kernel(float *d_V, float *d_H, float *d_W, int nh, int nw);
__global__ void conv4d_full_kernel(float *d_V, float *d_H, float *d_W, int nh, int nw);
__global__ void filter3d_valid_kernel(float *d_H, float *d_V, float *d_W, int nv, int nw);
__global__ void filter4d_valid_kernel(float *d_H, float *d_V, float *d_W, int nv, int nw);
__global__ void filter3d_full_kernel(float *d_V, float *d_H, float *d_W, int nh, int nw);
__global__ void filter4d_full_kernel(float *d_V, float *d_H, float *d_W, int nh, int nw);
__global__ void mapping_kernel(float *d_H, float *d_hb, int nh);
__global__ void sampling_kernel(float *d_H, float *d_HS, float *d_r, int C, int nh);
__global__ void pooling_kernel(float *d_P, float *d_HS, int C, int nh);
__global__ void prod3d_kernel(float *d_prod, float *d_V, float *d_H, int nv, int nh);
__global__ void sum_along_channels_kernel(float *d_act, float *d_batches, int imsize, int channels, int batchsize);
__global__ void sum_along_channels_new_kernel(float *d_act, float *d_batches, int imsize, int channels, int batchsize);

///////////////////// some utils //////////////////
__global__ void init_with_zero_kernel(float *d_A, int length);
__global__ void sigmoid_kernel(float *d_A, int length);
__global__ void copy_kernel(float *d_des, float *d_src, int length);
__global__ void iadd_num_kernel(float *d_A, float x, int length);
__global__ void isub_num_kernel(float *d_A, float x, int length);
__global__ void imul_num_kernel(float *d_A, float x, int length);
__global__ void idiv_num_kernel(float *d_A, float d, int length);
__global__ void iadd_vec_kernel(float *d_A, float *d_B, int length);
__global__ void isub_vec_kernel(float *d_A, float *d_B, int length);
__global__ void imul_vec_kernel(float *d_A, float *d_B, int length);
__global__ void idiv_vec_kernel(float *d_A, float *d_B, int length);
__global__ void add_kernel(float *d_ret, float *d_A, float *d_B, int length);
__global__ void sub_kernel(float *d_ret, float *d_A, float *d_B, int length);
__global__ void mul_kernel(float *d_ret, float *d_A, float *d_B, int length);
__global__ void div_kernel(float *d_ret, float *d_A, float *d_B, int length);
__global__ void sub_num_kernel(float *d_ret, float *d_A, float x, int length);
__global__ void add_num_kernel(float *d_ret, float *d_A, float x, int length);
__global__ void mul_num_kernel(float *d_ret, float *d_A, float x, int length);
__global__ void div_num_kernel(float *d_ret, float *d_A, float x, int length);

__global__ void truncate_kernel(float *d_A, float low, float high, int length);

