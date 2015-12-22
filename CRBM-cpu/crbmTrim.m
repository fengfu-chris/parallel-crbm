function [trimedImg] = crbmtrim(img, ws, C)

trimedImg = img;

if mod(size(img,1)-ws+1, C)~=0
	n = mod(size(img,1)-ws+1, C);
	trimedImg(1:floor(n/2), : ,:) = [];
	trimedImg(end-ceil(n/2)+1:end, : ,:) = [];
end

if mod(size(img,2)-ws+1, C)~=0
	n = mod(size(img,2)-ws+1, C);
	trimedImg(:, 1:floor(n/2), :) = [];
	trimedImg(:, end-ceil(n/2)+1:end, :) = [];
end

return