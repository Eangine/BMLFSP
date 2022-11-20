function preds = KNN(X,y, M, k, Xt)
% function preds = KNN(y, X, M, k, Xt)
% X :训练集
% y :训练集的标签
% M ：距离的度量矩阵
% k :knn的个数
% Xt:测试集

add1 = 0;
if (min(y) == 0)
    y = y + 1;
    add1 = 1;
end
[n,~] = size(X);
[nt,~] = size(Xt);
% when M id a SDP matrix,we can us kdtree in matlab
[~,val] = eig(M);
value = diag(val);
if min(value) < 1e-3
%     disp('M is not a SDP matrix, can not use kd-tree');
    K = (X*M*Xt');
    l = zeros(n,1);
    for i=1:n
        l(i) = (X(i,:)*M*X(i,:)');
    end
    lt = zeros(nt,1);
    for i=1:nt
        lt(i) = (Xt(i,:)*M*Xt(i,:)');
    end
    D = zeros(n, nt);
    for i=1:n
        for j=1:nt
            D(i,j) = l(i) + lt(j) - 2 * K(i, j);
        end
    end
    [~, Inds] = sort(D);
    preds = zeros(nt,1);
    for i=1:nt
        counts = [];
        for j=1:k
            if (y(Inds(j,i)) > length(counts))
                counts(y(Inds(j,i))) = 1;
            else
                counts(y(Inds(j,i))) = counts(y(Inds(j,i))) + 1;
            end
        end
        [~, preds(i)] = max(counts);
    end
    
else
    % use kd-tree
    if X > 2000
        KD = createns(X,'dist','mahalanobis','Cov',M);
        [idx, ~] = knnsearch(KD,Xt,'dist','mahalanobis','Cov',M,'k',k);
    else
        [idx, ~] = knnsearch(X,Xt,'dist','mahalanobis','Cov',M,'k',k);
    end
    preds = zeros(size(Xt,1),1); % 预测结果
    for i=1:size(Xt,1)
        counts = []; % 统计出现次数
        for j=1:k       
            if (y(idx(i,j)) > length(counts))
                counts(y(idx(i,j))) = 1;
            else
                counts(y(idx(i,j))) = counts(y(idx(i,j))) + 1;
            end
        end
        [~, preds(i)] = max(counts);
    end
end
if (add1 == 1)
    preds = preds - 1;
end