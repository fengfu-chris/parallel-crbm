function [crbm] = crbmTrain(patches, pars, W0, vb0, hb0)
% crbmTrain trains a crbm with inputs: patches and pars, 
% the output is saved in crbm structure which contains three fields
% W,vb and hb
% 
%  [crbm] = crbmTrain(patches, pars)
%
% patches: patchdim * numpatches
% pars: a structure contains  ws, Kin, Kout, epochs, epsilon, batchsize, save, savedir, etc
% crbm: a structure contains W, vb, hb

% disp(pars);

% set default parameters
if ~isfield(pars,'ws'), pars.ws = 8; end
if ~isfield(pars,'epoch'), pars.epoch = 10; end
if ~isfield(pars,'batchsize'), pars.batchsize = 1; end
if ~isfield(pars, 'save'),pars.save = 'true';end

ws        = pars.ws;
Kout      = pars.Kout;
Kin       = pars.Kin;
% epochs    = pars.maxEpoch;
epsilon   = pars.epsilon;
batchsize = pars.batchsize;

% initialize
%W = 0.01 * randn(ws^2, Kin, Kout);
if nargin  >= 3,
    W = W0;
    vb = vb0;
    hb = hb0;
else
    W = 0.1 * randn(ws^2, Kin, Kout);
    vb = 0 * ones(Kin, 1);
    hb = -0.1 * ones(Kout,1);
end

% fprintf('std of W = %.2f\n', std(W(:)));

Winc  = 0;   vbinc = 0;  hbinc = 0;

[patchdim, numpatches] = size(patches);
patchsize = sqrt(single(patchdim)/single(Kin));

% load IMAGES
for it = 1:pars.maxIter,
    momentum = 0.9 - 0.4 * (it <= 20000);  
   
    % batchStartId = floor(rand() * (numpatches - batchsize + 1)) + 1;
    batchStartId = mod((it-1)*pars.batchsize, numpatches - batchsize + 1) + 1;
    batch_V = patches(:, batchStartId : batchStartId + pars.batchsize - 1);
    batch_V = reshape(batch_V, [patchsize, patchsize, Kin, pars.batchsize]);
    
    [dW, dh, dv] = crbmUpdates(batch_V, W, hb, vb, pars);
    
    % update parameters
    Winc  = momentum * Winc  + epsilon * dW;
    vbinc = momentum * vbinc + epsilon * dv;
    hbinc = momentum * hbinc + epsilon * dh;
    
    W  = W  + Winc;
    vb = vb + vbinc;
    hb = hb + hbinc;
    
    if(mod(it, 100) == 0),
          fprintf('iter:  %d\n', it);
%         if(strcmp(pars.display, 'on') == 1),   
%             display_network(W);     
%         end
    end
end

crbm.W  = W;
crbm.vb = vb;
crbm.hb = hb;

end
