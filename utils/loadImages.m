function [images,flist] = loadImages(fpath, FMT, rows, cols)
% loadImages loads images from a specific path
% usage: [images] = loadImages(fpath, FMT, rows, cols)
% e.g.  
%  fpath = 'data/faces300/'
%  FMT = 'jpg'
%  rows = 120, cols = 100

if nargin < 3,
    rows = 100;
    cols = 100;
end

flist = dir(sprintf('%s/*.%s', fpath, FMT));
numImages = numel(flist);

images = zeros(rows, cols, numImages);

for imidx = 1:numImages,
	fprintf('[%2d]', imidx);
	fname = sprintf('%s/%s', fpath, flist(imidx).name);
	im = imread(fname);

	if size(im,3)>1
		im2 = double(rgb2gray(im));
	else
		im2 = double(im);
    end
    
    im2 = imresize(im2, [rows, cols], 'bicubic');

	imdata = whitenImage(im2);
	imdata = sqrt(0.1) * imdata;      % just for some trick??
    
    images(:,:,imidx) = imdata;
    
    if rem(imidx,25)==0,
        fprintf('\n');
    end
end

fprintf('\nSucceed!\n');

end
