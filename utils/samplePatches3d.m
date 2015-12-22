function patches = samplePatches3d(IMAGES, patchsize, numpatches)
% samplePatches3d sample patches for multichannels images
% usage: [patches] = samplePatches3d(images, patchsize, numpatches)
%
% for example:
%   images is of size height * width * numchannels * m 
%   patchsize = 30 ( = patchwidth = patchheight)
%   numpatches = 1000
    
%size of images
[height, width, Kin, numImages] = size(IMAGES);

patches = zeros(patchsize * patchsize * Kin, numpatches);

%indices of randomly picked images 
img_ind = unidrnd(numImages, [1, numpatches]); 

%row and column indices of randomly picked images
row_ind = unidrnd(height - patchsize + 1, [1, numpatches]);
col_ind = unidrnd(width  - patchsize + 1, [1, numpatches]);

%length of unrolled patches
patch_len = patchsize * patchsize * Kin;

for i = 1 : numpatches
    cur_patch = IMAGES(row_ind(i) : row_ind(i)+patchsize-1,...
                       col_ind(i) : col_ind(i)+patchsize-1,...
		               :,...
                       img_ind(i));
    patches(:,i) = reshape(cur_patch, [patch_len, 1]);
end

%% normalization
patches = normalizeData(patches);

end

%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
% pstd = 3 * std(patches(:));
% patches = max(min(patches, pstd), -pstd) / pstd;

end
