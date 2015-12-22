function [dW, dh, dv] = crbmUpdates(batch_V, W, hb, vb, pars)

ws = pars.ws;

%% START POSITIVE PHASE 
[~, batch_HP] = crbmInfer(batch_V, W, hb, pars);

posprods  = crbmVishid(batch_V, batch_HP, ws) / single(pars.batchsize);
posvisact = squeeze(mean(mean(mean(batch_V,1),2),4));
poshidact = squeeze(mean(mean(mean(batch_HP,1),2),4));

%% START NEGATIVE PHASE 
batch_Vneg = crbmRecon(batch_HP, W, vb, pars);
[~, batch_HPneg] = crbmInfer(batch_Vneg, W, hb, pars);

negprods  = crbmVishid(batch_Vneg, batch_HPneg, ws) / single(pars.batchsize);
negvisact = squeeze(mean(mean(mean(batch_Vneg,1),2),4));
neghidact = squeeze(mean(mean(mean(batch_HPneg,1),2),4));

%% UPDATE WEIGHTS AND BIASES 
dW1 = (posprods - negprods)/(size(batch_HP,1) * size(batch_HP,2));
dW2 =  - pars.l2reg * W; 
dW  = dW1 + dW2;

dh1 = poshidact - neghidact;
db_sp = poshidact - pars.pbias;
dh2 = - pars.pbias_lambda * db_sp;
dh = dh1 + dh2;

dv = (posvisact - negvisact);

return
