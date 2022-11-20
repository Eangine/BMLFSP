function [S,D] = construct_SD(data,label)
% Constructing similar constraint sets S and dissimilar constraint sets D
% data:  a n x d matrix
% label: a n x 1 vector
% return:  
%        S: a n x n matrix, if S(i,j)==1, it means that x_i and x_j have
%           the same label, else S(i,j)==0
%        D: a n x n matrix, if D(i,j)==1, it means that x_i and x_j have
%           the different label
[n, ~] = size(data);
[lablist, ~, labels] = unique(label);
nclass = length(lablist);
label_matrix = false(n, nclass);
label_matrix(sub2ind(size(label_matrix), (1:length(labels))', labels)) = true;
same_label = logical(double(label_matrix) * double(label_matrix'));

%1 Select 10 target neighbors, and construct the similar constraint sets S
no_targets = 10;         % number of target neighbors
sum_X = sum(data .^ 2, 2);
DD_same = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (data * data')));
DD_same(~same_label) = Inf; DD_same(1:n + 1:end) = Inf;
[~, targets_ind] = sort(DD_same, 2, 'ascend');
targets_ind = targets_ind(:,1:no_targets);
S = zeros(n, n);
S(sub2ind([n n], vec(repmat((1:n)', [1 no_targets])), vec(targets_ind))) = 1;

%2 Construct the disimilar constraint sets D and use the all pair constraints
no_targets = 10;         % number of target neighbors
DD_diff = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (data * data')));
DD_diff(same_label) = Inf; DD_diff(1:n + 1:end) = Inf;
[~, targets_ind] = sort(DD_diff, 2, 'ascend');
targets_ind = targets_ind(:,1:no_targets);
D = zeros(n, n);
D(sub2ind([n n], vec(repmat((1:n)', [1 no_targets])), vec(targets_ind))) = 1;
% D = ~same_label;
end




