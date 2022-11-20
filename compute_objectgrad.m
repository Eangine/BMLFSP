function [obj_value,delta_M] = compute_objectgrad(data,batchCurrent,spMatrix,S,D,M,param)
[~,d] = size(data);
obj_value = 0;
delta_M = zeros(d,d);
for idx = 1:length(batchCurrent)
    i = batchCurrent(idx);
    Si = find(S(i,:)==1);
    Di = find(D(i,:)==1);
    sim_value = 0;
    dis_value = 0;
    sim_grad = zeros(d,d);
    for j = 1:length(Si)
        dist_ij = (data(i,:) - data(Si(j),:)) * M * (data(i,:) - data(Si(j),:))';
        X_ij = (data(i,:) - data(Si(j),:))' * (data(i,:) - data(Si(j),:));
        sim_grad = sim_grad + exp(-param.gamma1 * dist_ij) * X_ij;
        sim_value = sim_value + exp(-param.gamma1 * dist_ij);
    end
    dis_grad = zeros(d,d);
    for l = 1:length(Di)
        dist_il = (data(i,:) - data(Di(l),:)) * M * (data(i,:) - data(Di(l),:))';
        X_il = (data(i,:) - data(Di(l),:))' * (data(i,:) - data(Di(l),:));
        dis_grad = dis_grad + exp(-param.gamma2 * dist_il) * X_il;
        dis_value = dis_value + exp(-param.gamma2 * dist_il);
    end
    Qs = -1/param.gamma1 * log(sim_value/length(Si));
    Qd =  1/param.gamma2 * log(dis_value/length(Di));
    if (Qs + Qd) > -inf
        obj_value = obj_value + (Qs+Qd)*spMatrix(i);
        if sim_value == 0 && dis_value ~= 0
            delta_M = delta_M - (dis_grad/dis_value)*spMatrix(i);
            disp("!");  
        end
        if dis_value == 0 && sim_value ~= 0
            delta_M = delta_M + (sim_grad/sim_value)*spMatrix(i);
            disp("!");
        end
        if sim_value == 0 && dis_value == 0
            error('The code is error');
        end
        if sim_value ~= 0 && dis_value ~= 0
            delta_M = delta_M + (sim_grad/sim_value-dis_grad/dis_value)*spMatrix(i);
        end
    else
        continue;
    end
end
I = eye(d);
delta_M = delta_M + 2*param.beta*(M);
obj_value = obj_value + param.beta*norm(M, "fro")^2;
end