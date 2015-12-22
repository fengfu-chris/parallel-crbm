% make_compile

%%
clc, clear; close all
addpath utils/
addpath CRBM-cpu/

Kout_list = [2, 4, 8, 16, 32, 64, 128, 256];
batchsize_list = [1, 2, 4, 8, 16, 32, 64];

Kout_list = Kout_list(6);
batchsize_list = batchsize_list(1);

%%
for i = 1:length(Kout_list),
    for j = 1:length(batchsize_list),
pars.nv = int32(31);
pars.nw = int32(8);
pars.nh = int32(pars.nv - pars.nw + 1);
pars.C = int32(2);
pars.Kin = int32(1);
pars.Kout = int32(Kout_list(i));
pars.epsilon = single(0.005);
pars.momemtum = single(0.5);
pars.l2reg = single(0.01);
pars.pbias = single(0.0015);
pars.pbiasL = single(5.0);
pars.std_gaussian = single(0.04);
pars.maxIter = int32(20000);
pars.SAVE_PER_ITERS = int32(1000);
pars.batchsize = int32(batchsize_list(j));
pars.numsamples = int32(10000); 
pars.Vtype = 'gaussian';
pars.Htype = 'gaussian';
pars.DEBUG = 'no';

%
load IMAGES
patches = single(samplePatches(IMAGES, double(pars.nv), double(pars.numsamples)));

W0 = single(0.1 * randn(pars.nw*pars.nw, pars.Kin, pars.Kout));
vb0 = single(0 * ones(pars.Kin, 1));
hb0 = single(-0.1 * ones(pars.Kout,1));

pars.ws = pars.nw;
pars.infer_type = pars.Htype;
pars.recon_type = pars.Vtype;
pars.pbias_lambda = pars.pbiasL;
tic,
[crbm] = crbmTrain(patches, pars, W0, vb0, hb0);
cputime = toc * 5; % Iterations: 50000 -> 10000, thus, the record time should be 5x
fprintf('kout = %d, bs = %d, cpu time: %f seconds. \n', pars.Kout, pars.batchsize, cputime);
W10 = crbm.W;
display_network(W10);

    end
end