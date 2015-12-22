function [prod] = crbmVishid(batch_V, batch_hp, ws)

Kin  = size(batch_V,3);
Kout = size(batch_hp,3);
batchsize = size(batch_V, 4);

selidx1 = size(batch_hp,1):-1:1;  selidx2 = size(batch_hp,2):-1:1;
prod = zeros(ws, ws, Kin, Kout);

for a = 1:batchsize,    
    for b = 1:Kout
        for c = 1:Kin
            prod(:,:,c,b) = prod(:,:,c,b) + conv2(batch_V(:,:,c,a), batch_hp(selidx1,selidx2,b,a), 'valid');
        end
    end
end
prod = reshape(prod, [ws^2,Kin,Kout]);

return
