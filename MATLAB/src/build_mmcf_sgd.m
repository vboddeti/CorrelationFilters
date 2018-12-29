% build_mmcf_sgd.m

function out = build_mmcf_sgd

global args;
global labels;
global data_freq;

global w;
global wBias;
global lambda;
global eta0;
global t;
global D;
global T;

if ~args.psd_flag
    compute_psd;
    args.psd_flag = 1;
end

dim = args.dim;
d = prod(args.fft_size);
alpha = args.alpha;
beta = args.beta;
ind = 1:(dim+1):dim^2;
T = beta*D;
T(:,ind) = alpha*ones(d,dim)+T(:,ind);

[m,n,dim,num] = size(data_freq);

lambda = 1/(args.C*num);
num_epochs = args.num_epochs;
data_freq = reshape(data_freq,[m*n*dim,num]);
ind = randperm(num);
data_freq = data_freq(:,ind);
labels = labels(ind);
eta0 = determineEta0(1,num);

t = 0;
wBias = 0;
w = zeros(m*n*dim,1);

for epoch = 1:num_epochs
    ind = randperm(num);
    data_freq = data_freq(:,ind);
    labels = labels(ind);
    train(1,num, eta0);
    [cost,loss,nerr] = test(1, num);
    disp([epoch cost loss nerr]);
end

out.filt_freq = reshape(w,[m,n,dim]);
out.filt = ifft2(out.filt_freq,'symmetric')*sqrt(prod(args.fft_size));
out.b = wBias;

data_freq = reshape(data_freq,[m,n,dim,num]);
[labels,ind] = sort(labels,'descend');
data_freq = data_freq(:,:,:,ind);
end

function w_norm = wnorm
global w;
global T;
global args;
global wBias;

dim = args.dim;
w = reshape(w,[prod(args.fft_size),dim]);
temp = fusion_matrix_multiply(T,w,[dim,dim],[dim,1]);
w_norm = sum(fusion_matrix_multiply(conj(w),temp,[1,dim],[dim,1]));
w_norm = w_norm + wBias*wBias;
w = reshape(w,[prod(args.fft_size)*dim,1 ]);
end

function [ploss,pnerr] = testOne(x,y)
global w;
global wBias;
s = real(w'*x)+ wBias;
ploss = max(0,1-s*y);
pnerr = (s*y <= 0);
end

function [] = trainOne(x,y,eta)
global w;
global T;
global args;
global wBias;
global lambda;
dim = args.dim;
s = real(w'*x) + wBias;
etab = eta * 0.01;
z = s*y;
if z > 1
    d = 0;
else
    d = y;
end
temp = fusion_matrix_multiply(T,reshape(w,[prod(args.fft_size),dim]),[dim,dim],[dim,1]);
temp = reshape(temp,[prod(args.fft_size)*args.dim,1]);
w = w - eta*lambda*temp;
if d ~=0 
    w = w + eta*d*x;
end
wBias = wBias*(1 - etab*lambda) + etab*d;
end

function [] = train(start, stop, eta0)
global t;
global data_freq;
global labels;
global lambda;
for i = start:stop
    eta = eta0/(1+lambda*eta0*t);
    trainOne(data_freq(:,i), labels(i), eta);
    t=t+1;
end
end

function [cost,loss,nerr] = test(start, stop)
global data_freq;
global labels;
global lambda;
nerr = 0;
loss = 0;
for i = start:stop
    [ploss,pnerr] = testOne(data_freq(:,i), labels(i));
    loss = loss + ploss;
    nerr = nerr + pnerr;
end
nerr = nerr/(stop - start + 1);
loss = loss / (stop - start + 1);
cost = loss + 0.5*lambda*wnorm;
end

function cost = evaluateEta(start, stop, eta)
global w;
global data_freq;
global labels;
global lambda;
global wBias;

wBias = 0;
w = zeros(size(data_freq,1),1);

for i = start:stop
    trainOne(data_freq(:,i),labels(i),eta);
end
loss = 0;
for i = start:stop
    [ploss,~] = testOne(data_freq(:,i),labels(i));
    loss = loss + ploss;
end
loss = loss/(stop-start+1);
cost = loss + 0.5*lambda*wnorm;
end

function eta0 = determineEta0(start, stop)
factor = 2.0;
loEta = 1;
loCost = evaluateEta(start, stop, loEta);
hiEta = loEta * factor;
hiCost = evaluateEta(start, stop, hiEta);
if (loCost < hiCost)
    while (loCost < hiCost)
        hiEta = loEta;
        hiCost = loCost;
        loEta = hiEta / factor;
        loCost = evaluateEta(start, stop, loEta);
    end
else if (hiCost < loCost)
        while (hiCost < loCost)
            loEta = hiEta;
            loCost = hiCost;
            hiEta = loEta*factor;
            hiCost = evaluateEta(start, stop, hiEta);
        end
    end
end
eta0 = loEta;
end