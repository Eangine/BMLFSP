function  [M] = bmlsp(data,label,param)
% data: a n x d matrix
% label: a n x 1 vector
% parameter: a parameter structure
[N,d] = size(data);
M = eye(d,d);
% 1 Gradient descent related parameter setting
stepSize = 1e-3;       % step size
max_iter = 100;        % maximum number of iterations
% 2 Constructing S and D sets
[S,D] = construct_SD(data,label);
spMatrix = zeros(N,1);
SPCounter = 0;
LossRecord = [];
BatchLossRecord = [];
[~,loss_list] = compute_SP_Matrix(data,S,D,M,param);
[~,n] = sort(loss_list, 'ascend');
alpha_rate = 0.5;
param.alpha = loss_list(n(ceil(N*alpha_rate)));
if param.alpha<0
    param.alpha = 0.001;
end
while sum(spMatrix~=0)/N < param.rate
    [spMatrix] = compute_SP_Matrix(data,S,D,M,param);
    % 3 Perform projection gradient descent
    iters = 1;
    BatchLoss = inf;
    counter = 1;
    while iters < max_iter
        [batchCell] = gengerateMiniBatch(spMatrix);
        M_old = M;
        BatchLoss_old = BatchLoss;
        BatchLoss = 0;
        for batchId = 1:length(batchCell)
            batchCurrent = batchCell{batchId};
            [obj,delta_M] = compute_objectgrad(data,batchCurrent,spMatrix,S,D,M,param);
            counter = counter + 1;
            M = M - stepSize * delta_M;
            try
                M = (M + M')/2;  % enforce A to be symmetric
                [V,L] = eig(M);  % V is an othornomal matrix of M's eigenvectors, 
                                 % L is the diagnal matrix of M's eigenvalues, 
                L = max(L, 0);
                M = V*L*V';
            catch
                M = eye(d,d);
            end
            BatchLoss = BatchLoss + obj;
            LossRecord = [LossRecord obj];
        end
        BatchLossRecord = [BatchLossRecord BatchLoss];
        
        iters = iters + 1;
        if norm(M_old-M,'fro') < 1e-3 ||...
                abs(BatchLoss_old - BatchLoss)/abs(max(BatchLoss_old,BatchLoss)) < 1e-3
            M = M_old;
            break;
        end
        
        if BatchLoss < BatchLoss_old
            stepSize = stepSize * 1.05;
        else
            stepSize = stepSize / 2;
        end
        
    end
    if sum(spMatrix) == 0 || SPCounter >= 100
        print("stop early !!!")
        break;
    end
    param.alpha = param.alpha * param.gamma;
    SPCounter = SPCounter + 1;
end
% disp(["SPCounter:",num2str(SPCounter)]);

if ~issymmetric(M)
    M = (M+M')/2;
end

end

