function [batch_hs, batch_hp] = crbmInfer(batch_imgs, W, hb, pars)

%% get accumulate acctivation
ws = sqrt(size(W,1));
Kin = size(W,2);
Kout = size(W,3);

batch_imgs = reshape(batch_imgs, [pars.nv, pars.nv, pars.Kin, pars.batchsize]);

batch_Z = zeros([size(batch_imgs,1)-ws+1, size(batch_imgs,2)-ws+1, Kout, pars.batchsize]);
batch_hs = zeros(size(batch_Z));  batch_hp = zeros(size(batch_Z));

infer_type = pars.infer_type;
std_gaussian = pars.std_gaussian;
% WW = reshape(W(end:-1:1, :, :), [ws, ws, size(W, 2), size(W,3)]);
% parfor a = 1:pars.batchsize,
for a = 1:pars.batchsize,
    for b = 1:Kout
        for c = 1:Kin
            W_temp = reshape(W(end:-1:1, c, b), [ws,ws]);
            % W = WW(:,:,c,b);
            batch_Z(:,:,b, a) = batch_Z(:,:,b,a) + conv2(batch_imgs(:,:,c,a), W_temp, 'valid');
        end
        
        if strcmp(infer_type, 'gaussian'),
            batch_Z(:,:,b,a) = 1/std_gaussian .* (batch_Z(:,:,b,a) + hb(b));
        else
            batch_Z(:,:,b,a) = batch_Z(:,:,b,a) + hb(b);
        end
    end
end


% %% sampling from activation
 C = pars.C;

for a = 1:pars.batchsize,
    Z = batch_Z(:,:,:,a);
    I = exp(Z);
    I_mult = zeros(C^2+1, numel(I) / C^2);
    I_mult(end,:) = 1;
    
    for c = 1:C
        for r = 1:C
            temp = I(r:C:end, c:C:end, :);
            I_mult( (c-1)*C+r, :) = temp(:);
        end
    end
    
    %%
    % H: sampled data from posprobs distribution
    % HP: posprobs of hidden units
    
    % hidden layer
    sumI = sum(I_mult);
    HP = I_mult./repmat(sumI, [size(I_mult,1),1]);
    
    cumHP = cumsum(HP);
    unifrnd = rand(1,size(HP,2));
    temp = cumHP > repmat(unifrnd,[size(HP,1),1]);
    Hindx = diff(temp,1);
    
    H = zeros(size(HP));
    H(1,:) = 1-sum(Hindx);
    H(2:end,:) = Hindx;
    
    % convert back to original sized matrix
    for c = 1:C
        for r = 1:C
            batch_hs(r:C:end, c:C:end, :, a) = reshape(H((c-1)*C + r,:),  ...
                [size(batch_hs,1)/C, size(batch_hs,2)/C, size(batch_hs,3)]);
            batch_hp(r:C:end, c:C:end, :, a) = reshape(HP((c-1)*C + r,:), ...
                [size(batch_hs,1)/C,size(batch_hs,2)/C, size(batch_hp,3)]);
        end
    end
end
return
