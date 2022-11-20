function [spMatrix,loss_list] = compute_SP_Matrix(data,S,D,M,param)
    [n,~] = size(data);
    spMatrix = zeros(n,1);
    loss_list = zeros(n,1);
    for i = 1:n
        Si = find(S(i,:)==1);
        Di = find(D(i,:)==1);
        sim_value = 0;
        dis_value = 0;
        for j = 1:length(Si)
            dist_ij = (data(i,:) - data(Si(j),:)) * M * (data(i,:) - data(Si(j),:))';
            sim_value = sim_value + exp(-param.gamma1*dist_ij);
        end
        for k = 1:length(Di)
            dist_ik = (data(i,:) - data(Di(k),:)) * M * (data(i,:) - data(Di(k),:))';
            dis_value = dis_value + exp(-param.gamma2*dist_ik);
        end
        Qs = -1/param.gamma1 * log(sim_value/length(Si));
        Qd =  1/param.gamma2 * log(dis_value/length(Di));
        loss_i = Qs + Qd;
        loss_list(i,1) = loss_i;
        if loss_i < param.alpha*(1/param.q-1)
            spMatrix(i) = 1;
        elseif loss_i > param.alpha/param.q
            spMatrix(i) = 0;
        else
            spMatrix(i) = ((1/param.q)-(loss_i/param.alpha))^(1/(param.p-1));
        end
    end
end
