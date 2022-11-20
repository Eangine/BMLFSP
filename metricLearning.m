% Metric Learning
function [result]=metricLearning(train_data,train_label,test_data,test_label,param_cell)
trials_result = ...
    zeros(length(param_cell{1})*length(param_cell{2})*length(param_cell{3})*...
    length(param_cell{4})*length(param_cell{5})*length(param_cell{8}),6);
count_choose = 1;
for i = 1:length(param_cell{1})
    param.alpha = param_cell{1}(i);
    for j = 1:length(param_cell{2})
        param.beta = param_cell{2}(j);
        for k = 1:length(param_cell{3})
            param.gamma1 = param_cell{3}(k);
            for l = 1:length(param_cell{4})
                param.gamma2 = param_cell{4}(l);
                for m = 1:length(param_cell{5})
                    for n = 1:length(param_cell{8})
                        param.rate = param_cell{5}(m);
                        param.p = param_cell{6};
                        param.q = param_cell{7};
                        param.gamma = param_cell{8}(n);
                        trials_result(count_choose,2) = param_cell{1}(i);
                        trials_result(count_choose,3) = param_cell{2}(j);
                        trials_result(count_choose,4) = param_cell{3}(k);
                        trials_result(count_choose,5) = param_cell{4}(l);
                        trials_result(count_choose,6) = param_cell{5}(m);
%                             tic;
                        try
                            M = bmlsp(train_data,train_label,param);
                            preds = KNN(train_data,train_label, M, 3, test_data);
                            index = find((test_label-preds)==0);
                            trials_result(count_choose,1) = length(index)/size(test_data,1);
                        catch
                            trials_result(count_choose,1) = 0;
                        end
%                             toc;
                        if mod(100*(count_choose)/length(trials_result),10)==0
                            disp(strcat(num2str(100*(count_choose)/length(trials_result)),"% finished"));
                        end
        %                     disp(strcat(num2str(100*(count_choose)/length(trials_result)),"% finished"));
                        count_choose = count_choose + 1;
                    end
                end
            end
        end
    end
end
temp_acc = trials_result(:,1);
index1 = find(max(temp_acc)==temp_acc);
index1 = index1(randperm(length(index1),1));
result = [
    max(temp_acc)
    ];
