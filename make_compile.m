clear all

%% compile objects
system('rm CRBM-gpu/*.o');

cd CRBM-gpu/

system('nvcc -c -Xcompiler -fPIC utils.cu');
system('nvcc -c -Xcompiler -fPIC kernels.cu');
system('nvcc -c -Xcompiler -fPIC crbm.cu');
system('nvcc -c -Xcompiler -fPIC crbm_proxy.cu');

%% linking 
system('rm *.mexa64');

mex crbmTrainCUDA.cpp crbm.o kernels.o utils.o crbm_proxy.o -lcudart -lcurand -L/usr/local/cuda/lib64
mex crbmInferCUDA.cpp crbm.o kernels.o utils.o crbm_proxy.o -lcudart -lcurand -L/usr/local/cuda/lib64
mex crbmReconCUDA.cpp crbm.o kernels.o utils.o crbm_proxy.o -lcudart -lcurand -L/usr/local/cuda/lib64
mex computeActCUDA.cpp crbm.o kernels.o utils.o crbm_proxy.o -lcudart -lcurand -L/usr/local/cuda/lib64
mex computeVisHidCUDA.cpp crbm.o kernels.o utils.o crbm_proxy.o -lcudart -lcurand -L/usr/local/cuda/lib64
mex crbmUpdatesCUDA.cpp crbm.o kernels.o utils.o crbm_proxy.o -lcudart -lcurand -L/usr/local/cuda/lib64

system('rm *.o')
cd ..
