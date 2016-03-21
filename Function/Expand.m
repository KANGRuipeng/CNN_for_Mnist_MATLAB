function B = Expand(A, S)

%Build the map first, finally fill the number

if nargin < 2
    error('Size vector must be provided.  See help.');
end

SA = size(A);  % Get the size (and number of dimensions) of input.

if length(SA) ~= length(S)
   error('Length of size vector must equal ndims(A).  See help.')
elseif any(S ~= floor(S))
   error('The size vector must contain integers only.  See help.')
end

T = cell(length(SA), 1);
for ii = length(SA) : -1 : 1
    H = zeros(SA(ii) * S(ii), 1);   %  One index vector into A for each dim.
    H(1 : S(ii) : SA(ii) * S(ii)) = 1;   %  Put ones in correct places.
    T{ii} = cumsum(H);   %  Cumsumming creates the correct order.
end

B = A(T{:});   %  Feed the indices into A.