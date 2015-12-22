function [features2, maxhidact2, maxactcenter2] = getFeatures2(imdata, crbm1, crbm2, pars1, pars2) 

ws1 = sqrt(size(crbm1.W, 1));
ws2 = sqrt(size(crbm2.W,1));
C = pars1.C;

%% stage 1

imdata = crbmTrim(imdata, pars1.ws, pars1.C);
[ps] = crbmInfer2(imdata, crbm1.W, crbm1.hb, pars1);

% [hs2, hp2] = crbmInfer(pp, crbm2.W, crbm2.hb, pars2);
% [poshidstates_hier,poshidprobs_hier] = crbm_inference_hierarchical(imdata,H2probs,crbm1,crbm2,pars1,pars2);
% [negdata] = crbm_reconstruct(poshidstates_hier, crbm1.W, crbm1.vbias);
negdata = imdata;

%% stage 2
ps = crbmTrim(ps, pars2.ws, pars2.C);
[hs2, hp2] = crbmInfer(ps, crbm2.W, crbm2.hb, pars2);

feature2_width = ws2 * C + ws1 - 1;
numF2 = size(hp2,3);
features2 = zeros(feature2_width, feature2_width, numF2);

maxhidact2 = zeros(1, numF2);
maxactcenter2 = zeros(2, numF2);

for i = 1:numF2
	hidtemp2 = hp2(:,:,i);
	maxhidact2(i) = max(hidtemp2(:));
	[maxactrow, maxactcol] = find(hidtemp2 == maxhidact2(i));
	indexP_row = maxactrow : maxactrow+ws2-1;
	indexP_col = maxactcol : maxactcol+ws2-1;

	indexH_row = (indexP_row(1)-1)*C+1 : indexP_row(end)*C;
	indexH_col = (indexP_col(1)-1)*C+1 : indexP_col(end)*C;

	indexV_row = indexH_row(1) : indexH_row(end)+ws1-1;
	indexV_col = indexH_col(1) : indexH_col(end)+ws1-1;

	maxactcenter2(:,i) = [mean(indexV_row); mean(indexV_col)];

	features2(:,:,i) = negdata(indexV_row, indexV_col);
end

return