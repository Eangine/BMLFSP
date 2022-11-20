clear;clc;

rootPath = '\';
dataPath = '\';
savePath = '\';
cd(dataPath);
files = dir([dataPath,'*.mat']);
cd(rootPath);
for idx = 1:length(files)
cd(dataPath);
load(files(idx).name);
disp(files(idx).name);
data = X;
label = Y;
cd(rootPath);
[n_sam] = length(label);
format long;
% 1 The parameters involved and the adjustment range

alpha = 1;             % the parameters of the regular-term alpha
beta = 10.^(-3:2:3);   % the parameters of the regular-term beta
gamma1 = -2.^(-4:2:4); % controls the number of similar samples at the farthest distance
gamma2 = 2.^(-4:2:4);  % controls the number of dissimilar samples at the nearest distance
rate = [0.96,0.98,1];
p = 2;
q = 0.5;
gamma = 1.5;

param_cell{1} = alpha;
param_cell{2} = beta;
param_cell{3} = gamma1;
param_cell{4} = gamma2;
param_cell{5} = rate;
param_cell{6} = p;
param_cell{7} = q;
param_cell{8} = gamma;
 
% 2 Product n_trials trials
n_trials = 4;
all_trials_result = zeros(n_trials,1);
parfor i_trials = 1:n_trials
    tic;
    disp(['i_trials = ',num2str(i_trials)]);
    % 3 Divide the train set and test set, and normalize the dataset
    ntrain = floor(n_sam*0.7);
    rand('state',sum(100*(i_trials+clock)));
    rand_idx = randperm(n_sam);
    train_index = rand_idx(1:ntrain);
    test_index = rand_idx(ntrain+1:end);
    train_data = data(train_index(:),:);
    train_label = label(train_index(:));
    test_data = data(test_index(:),:);
    test_label = label(test_index(:));
    % 3.1 normalize the train data
    [train_data,MU,SIGMA] = zscore(train_data); 
    % 3.2 normalize the test data and to prevent the singular values when normalizing
    for i = 1:size(test_data,1)
        test_data(i,:) = (test_data(i,:)-MU)./SIGMA; 
    end
    index = find(SIGMA == 0);
    test_data(:,index(:)) = 0;
    % 4 Learn for each set of parameters and choose the optimal one
    [result] = metricLearning(train_data,train_label,test_data,test_label,param_cell);
    all_trials_result(i_trials,1) = result'; 
    toc;
end
cd(savePath);
save([files(idx).name],'all_trials_result');
mean_acc = mean(all_trials_result(:,1));
std_acc = std(all_trials_result(:,1));
disp(strcat(files(idx).name(1:end-4),"  ","mean_acc=",num2str(mean_acc),"  ","std_acc=",num2str(std_acc)))
cd(rootPath);
end
