function [h, array] = display_network_new(A, display)
%
% [h,array] = display_network_new(A)
%
% Inputs:
%   --> A: [rows,cols,K] matrix
%

if nargin < 2,
    display = 'on';
end

warning off all

colormap(gray);

buf=1;

%% compute rows, cols
[rows cols K]=size(A);
if floor(sqrt(K))^2 ~= K
    n = ceil(sqrt(K));
    while mod(K, n)~=0 && n<1.2*sqrt(K),
		n = n+1;
	end
    m = ceil(K/n);
else
    n = sqrt(K);
    m = n;
end

array = -ones(buf+m*(rows+buf),buf+n*(cols+buf));

k=1;
for i=1:m
    for j=1:n
        if k>K, continue;  end

        clim = max(max(abs(A(:,:,k))));
        array(buf+(i-1)*(rows+buf)+(1:rows),buf+(j-1)*(cols+buf)+(1:cols)) = A(:,:,k)/clim;
        k = k+1;
    end
end

h = imagesc(array,'EraseMode','none',[-1 1]);
axis image off

% if strcmp(display,'off')
%     figure('visible','off');
%     imshow(array);
% else
%     figure();
%     imshow(array);
% end

warning on all
return
