function [batch_negdata] = crbmRecon(batch_hs, W, vb, pars)

ws  = sqrt(size(W,1));
Kin = size(W,2);
Kout = size(W,3);

batch_hs = reshape(batch_hs, [pars.nh, pars.nh, pars.Kout, pars.batchsize]);

batch_negdata = zeros(size(batch_hs,1) + ws - 1, size(batch_hs,2) + ws - 1, Kin, pars.batchsize);

recon_type = pars.recon_type;
% parfor a = 1:pars.batchsize,
for a = 1:pars.batchsize,
    for c = 1:Kin,
        for b = 1:Kout,
            H = reshape(W(:,c,b), [ws, ws]);
            batch_negdata(:,:,c,a) = batch_negdata(:,:,c,a) + conv2(batch_hs(:,:,b,a), H, 'full');
        end
        if strcmp(recon_type, 'gaussian'),
            batch_negdata(:,:,c,a) = batch_negdata(:,:,c,a) + vb(c);
        else
            batch_negdata(:,:,c,a) = sigmoid(batch_negdata(:,:,c,a) + vb(c));
        end
    end
end

return
