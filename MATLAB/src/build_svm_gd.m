% build_svm_gd.m

function out = build_svm_gd

global args;
global data;
global labels;

global w;
global wBias;
global a;
global aBias;
global lambda;
global eta0;
global t;
global tstart;
global K;

K = args.K;
tstart = args.tstart;
[m,n,dim,num] = size(data);
d = m*n;

lambda = 1/(num*args.C);
num_epochs = args.num_epochs;
data = reshape(data,[m*n*dim,num]);
ind = randperm(num);
data = data(:,ind);
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
    data = data(:,ind);
    labels = labels(ind);
    train(1,num, eta0, mu0);
    [cost,loss,nerr] = test(1, num);
    disp([epoch cost loss nerr]);
end

out.filt = reshape(a,[m,n,dim]);
out.b = aBias;

data = reshape(data,[m,n,dim,num]);
[labels,ind] = sort(labels,'descend');
data = data(:,:,:,ind);
end

function w_norm = wnorm
global w;
global wBias;

w_norm = w'*w;
w_norm = w_norm + wBias*wBias;
end

function a_norm = anorm
global a;
global aBias;

a_norm = a'*a;
a_norm = a_norm + aBias*aBias;
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
global args;
global wBias;
global aBias
global lambda;
K = args.K;

s = real(x'*w) + wBias;
etab = eta * 0.01;
z = s.*y;
d = ((z < 1).*y)/K;
w = w - eta*lambda*w;
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
global data;
global labels;
global lambda;
global tstart;
global K;
for i = start:K:stop
    eta = eta0/(1+lambda*eta0*t)^(args.exponent);
%     eta = 1/(lambda*(t+1));
    if t < tstart
        mu = 1;
    else
        mu = mu0/(1+mu0*(t - tstart));
    end
    imin = i;
    imax = min(i+K-1,stop);
    trainBatch(data(:,imin:imax), labels(imin:imax), eta, mu);
    t=t+1;
end
end

function [cost,loss,nerr] = test(start, stop)
global data;
global labels;
global lambda;
[loss,nerr] = testBatch(data, labels);
nerr = nerr/(stop - start + 1);
loss = loss / (stop - start + 1);
cost = loss + 0.5*lambda*anorm;
end

function cost = evaluateEta(start, stop, eta)
global a;
global w;
global data;
global labels;
global lambda;
global wBias;
global aBias;
global K;

wBias = 0;
w = zeros(size(data,1),1);
aBias = 0;
a = zeros(size(data,1),1);

for i = start:K:stop
    imin = i;
    imax = min(i+K-1,stop);
    trainBatch(data(:,imin:imax),labels(imin:imax),eta,1);
end
loss = testBatch(data,labels);
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