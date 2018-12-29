% build_svm_online.m

function out = build_svm_online

global args;
global labels;
global data;

[m,n,dim,num] = size(data);

num_pos = sum(labels==1);
num_neg = sum(labels==-1);

lambda = 1/(args.C*num);
pos_batch_size = min(0.2*num_pos,20);
neg_batch_size = min(0.2*num_neg,20);
K = pos_batch_size+neg_batch_size;

if num_pos <= num_neg
    num_val = num_neg;
    batch_size = neg_batch_size;
else
    num_val = num_pos;
    batch_size = pos_batch_size;
end

pos_idx = find(labels==1);
neg_idx = find(labels==-1);
data_freq = reshape(data_freq,[m*n*dim,num]);
w_freq = H;
w_freq = reshape(w_freq,[m*n*dim,1]);

num_epochs = 2;
pos_ind = 1;
neg_ind = 1;

num_iter = ceil(num_val/batch_size);
err = zeros(num_epochs*num_iter,1);

index_pos = randperm(num_pos);
index_neg = randperm(num_neg);

t = 1;
for i = 1:num_epochs
    for j = 1:num_iter
        
        w_old_freq = w_freq;
        
        if mod(t,100) == 0
            H = reshape(w_freq,[m*n,dim]);
            temp = fusion_matrix_multiply(S,H,[dim,dim],[dim,1]);
            val = sum(fusion_matrix_multiply(conj(H),temp,[1,dim],[dim,1]));
            err(t,1) = real(lambda*val/2+mean(sign(data_freq'*w_freq + b_freq)-labels));
            display(['iteration = ' num2str(t) ' Relative Error = ' num2str(err(t,1))]);
        end
        if pos_ind > num_pos
            pos_ind = mod(pos_ind,num_pos)+1;
            index_pos = randperm(num_pos);
        end
        if neg_ind > num_neg
            neg_ind = mod(neg_ind,num_neg)+1;
            index_neg = randperm(num_neg);
        end
        
        b_freq = mean(labels - data_freq'*w_freq);
        idx1 = pos_idx(index_pos(pos_ind:min(pos_ind+pos_batch_size,num_pos)));
        idx2 = neg_idx(index_neg(neg_ind:min(neg_ind+neg_batch_size,num_neg)));
        idx = [idx1;idx2];
        
        At = data_freq(:,idx);
        yt = labels(idx);
        idx1 = (At'*w_freq+b_freq).*yt<1;
        etat = 1/(lambda*t);
        
        tmp = reshape(w_freq,[m*n,dim]);
        tmp = fusion_matrix_multiply(S,tmp,[dim,dim],[dim,1]);
        tmp = reshape(tmp,[m*n*dim,1]);
        
        w1 = w_freq - etat*lambda*tmp + (etat/K)*sum(At(:,idx1).*repmat(yt(idx1)',[size(At,1),1]),2);
        
        w1 = reshape(w1,[m,n,dim]);
        w1 = h_prox(w1,args);
        w1 = reshape(w1,[m*n*dim,1]);
        
        H = reshape(w1,[m*n,dim]);
        temp = fusion_matrix_multiply(S,H,[dim,dim],[dim,1]);
        val = sum(fusion_matrix_multiply(conj(H),temp,[1,dim],[dim,1]));
        w1 = min(1,1/(sqrt(lambda)*val))*w1;
        w_freq = (t*w_freq + w1)/(t+1);
        
        pos_ind = pos_ind + pos_batch_size;
        neg_ind = neg_ind + neg_batch_size;
        t = t + 1;
    end
end

err = err(1:t,1);
if(t<maxIter)
    fprintf('\nW converged in %d iterations.',t);
else
    fprintf('\nW not converged in %d iterations.',maxIter);
end
w_avg = w_avg/max(1,t);

w1 = w_avg;
% w1 = w;

b = mean(labels-data'*w1);
scores = data'*w1+b;
[X,Y,T,AUC] = perfcurve(labels,scores,1);
fprintf('\n Online SVM: Accuracy on Training set = %.4f %%\n', 100*AUC);

data = reshape(data,[m,n,dim,num]);
out.filt = reshape(w1,[m,n,dim]);
out.b = b;
out.args = args;