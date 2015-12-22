function patches = samplePatches(IMAGES, patchsize, numpatches)
% sample patches from IMAGES
% Returns 10000 patches for training

if nargin < 1,
    addpath ../data/
    load IMAGES
    patchsize  = 8;  % we'll use 8x8 patches 
    numpatches = 10000;
end
    
% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchsize * patchsize, numpatches);

%size of images
[height, width, numImages] = size(IMAGES);

%indices of randomly picked images 
img_ind = unidrnd(numImages, [1, numpatches]); 

%row and column indices of randomly picked images
row_ind = unidrnd(height - patchsize + 1, [1, numpatches]);
col_ind = unidrnd(width  - patchsize + 1, [1, numpatches]);

%length of unrolled patches
patch_len = patchsize * patchsize;

for i = 1 : numpatches
    cur_patch = IMAGES(row_ind(i) : row_ind(i)+patchsize-1,...
                       col_ind(i) : col_ind(i)+patchsize-1,...
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
