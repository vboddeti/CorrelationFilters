% build_mmcf_asgd.m

function out = build_mmcf_asgd

global args;
global labels;
global data_freq;

global w;
global wBias;
global a;
global aBias;
global lambda;
global eta0;
global t;
global D;
global T;
global tstart;
global K;

if ~args.psd_flag
    compute_psd;
    args.psd_flag = 1;
end

K = args.K;
tstart = args.tstart;
dim = args.dim;
d = prod(args.fft_size);
alpha = args.alpha;
beta = args.beta;
ind = 1:(dim+1):dim^2;
T = beta*D;
T(:,ind) = alpha*ones(d,dim)+T(:,ind);

[m,n,dim,num] = size(data_freq);

lambda = args.C;
num_epochs = args.num_epochs;
data_freq = reshape(data_freq,[m*n*dim,num]);
ind = randperm(num);
data_freq = data_freq(:,ind);
labels = labels(ind);
eta0 = determineEta0(1,num);

t = 0;
wBias = 0;
w = reshape(args.w_init,[m*n*dim,1]);
aBias = 0;
a = reshape(args.w_init,[m*n*dim,1]);
mu0 = 1;

for epoch = 1:num_epochs
    ind = randperm(num);
    data_freq = data_freq(:,ind);
    labels = labels(ind);
    train(1,num, eta0, mu0);
    [cost,loss,nerr] = test(1, num);
    disp([epoch cost loss nerr]);
end

out.filt_freq = reshape(a,[m,n,dim]);
out.filt = ifft2(out.filt_freq,'symmetric')*sqrt(prod(args.fft_size));
out.b = aBias;

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
w_norm = real(w_norm);
w = reshape(w,[prod(args.fft_size)*dim,1 ]);
end

function a_norm = anorm
global a;
global T;
global args;
global aBias;

dim = args.dim;
a = reshape(a,[prod(args.fft_size),dim]);
temp = fusion_matrix_multiply(T,a,[dim,dim],[dim,1]);
a_norm = sum(fusion_matrix_multiply(conj(a),temp,[1,dim],[dim,1]));
a_norm = a_norm + aBias*aBias;
a_norm = real(a_norm);
a = reshape(a,[prod(args.fft_size)*dim,1]);
end

function [ploss,pnerr] = testBatch(x,y)
global a;
global aBias;
s = real(x'*a)+ aBias;
z = s.*y;
ploss = sum(max(0,1-z));
pnerr = sum(z <= 0);
end

function [] = trainBatch(x,y,eta,mu)
global a;
global w;
global T;
global args;
global wBias;
global aBias
global lambda;
dim = args.dim;
K = args.K;

s = real(x'*w) + wBias;
etab = eta * 0.01;
z = s.*y;
d = ((z < 1).*y)/K;
temp = fusion_matrix_multiply(T,reshape(w,[prod(args.fft_size),dim]),[dim,dim],[dim,1]);
temp = reshape(temp,[prod(args.fft_size)*args.dim,1]);
w = w - eta*lambda*temp;
for i = 1:length(d)
    if d(i) ~=0
        w = w + eta*d(i)*x(:,i);
    end
end
wBias = wBias*(1 - etab*lambda) + etab*sum(d);
a = a + mu*(w-a);
aBias = aBias + mu*(wBias - aBias);
end

function [] = train(start, stop, eta0, mu0)
global t;
global args;
global data_freq;
global labels;
global lambda;
global tstart;
global K;
for i = start:K:stop
    eta = eta0/(1+lambda*eta0*t)^(args.exponent);
    if t < tstart
        mu = 1;
    else
        mu = mu0/(1+mu0*(t - tstart));
    end
    imin = i;
    imax = min(i+K-1,stop);
    trainBatch(data_freq(:,imin:imax), labels(imin:imax), eta, mu);
    t=t+1;
end
end

function [cost,loss,nerr] = test(start, stop)
global data_freq;
global labels;
global lambda;
[loss,nerr] = testBatch(data_freq, labels);
nerr = nerr/(stop - start + 1);
loss = loss / (stop - start + 1);
cost = loss + 0.5*lambda*anorm;
end

function cost = evaluateEta(start, stop, eta)
global a;
global w;
global data_freq;
global labels;
global lambda;
global wBias;
global aBias;
global K;

wBias = 0;
w = zeros(size(data_freq,1),1);
aBias = 0;
a = zeros(size(data_freq,1),1);

for i = start:K:stop
    imin = i;
    imax = min(i+K-1,stop);
    trainBatch(data_freq(:,imin:imax),labels(imin:imax),eta,1);
end
loss = testBatch(data_freq,labels);
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