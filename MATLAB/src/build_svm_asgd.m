% build_svm_asgd.m

function out = build_svm_asgd

global args;
global data;
global labels;

global w;
global wBias;
global lambda;
global wDivisor;
global t;
global tstart;
global a;
global aDivisor;
global wFraction;
global aBias;

[m,n,dim,num] = size(data);

tstart = args.tstart;
lambda = 1/(args.C*num);
num_epochs = args.num_epochs;
data = reshape(data,[m*n*dim,num]);
ind = randperm(num);
data = data(:,ind);
labels = labels(ind,1);
eta0 = determineEta0(1,num);

t=0;
wDivisor = 1;
w = zeros(m*n*dim,1);
wBias = 0;
mu0 = 1;
a = zeros(m*n*dim,1);
aDivisor = 1;
wFraction = 0;
aBias = 0;

for epoch = 1:num_epochs
    ind = randperm(num);
    data = data(:,ind);
    labels = labels(ind,1);
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

function [] = renorm
global w;
global a;
global wDivisor;
global aDivisor;
global wFraction;

if (wDivisor ~= 1) || (aDivisor ~= 1) || (wFraction ~= 0)
    a = a/aDivisor + w*wFraction/aDivisor;
    w = w/wDivisor;
    wDivisor = 1;
    aDivisor = 1;
    wFraction = 0;
end
end

function w_norm = wnorm
global w;
global wBias;
global wDivisor;
w_norm = (w'*w)/(wDivisor*wDivisor);
w_norm = w_norm + wBias*wBias;
end

function a_norm = anorm
global a;
global aBias;
renorm;
a_norm = (a'*a);
a_norm = a_norm + aBias*aBias;
end

function [ploss,pnerr] = testOne(x,y)
global w;
global a;
global aBias;
global aDivisor;
global wFraction;
s = (a'*x);
if (wFraction ~=0)
    s = s + w'*x*wFraction;
end
s = s/aDivisor + aBias;
ploss = max(0,1-s*y);
pnerr = (s*y <= 0);
end

function [] = trainOne(x,y,eta,mu)
global w;
global a;
global wBias;
global wDivisor;
global aBias;
global aDivisor;
global wFraction;
global lambda;

if (wDivisor > 1e5) || (aDivisor > 1e5)
    renorm;
end
s = (w'*x)/wDivisor + wBias;
wDivisor = wDivisor/(1-eta*lambda);
z = s*y;
if z > 1
    d = 0;
else
    d = y;
end
etd = eta*d*wDivisor;
if (etd ~= 0)
    w = w+etd*x;
end

if (mu >= 1)
    a = a*0;
    aDivisor = wDivisor;
    wFraction = 1;
elseif (mu > 0)
    if (etd ~= 0)
        a = a - wFraction*etd*x;
    end
    aDivisor = aDivisor/(1 - mu);
    wFraction = wFraction + mu*aDivisor/wDivisor;
end

etab = eta * 0.01;
wBias = wBias*(1 - etab*lambda);
wBias = wBias + etab*d;
aBias = aBias + mu*(wBias - aBias);

end

function [] = train(start, stop, eta0, mu0)
global t;
global data;
global labels;
global lambda;
global tstart;
for i = start:stop
    eta = eta0/(1+lambda*eta0*t)^(0.75);
    if t < tstart
        mu = 1;
    else
        mu = mu0/(1+mu0*(t - tstart));
    end
    trainOne(data(:,i), labels(i), eta, mu);
    t=t+1;
end
renorm;
end

function [cost,loss,nerr] = test(start, stop)
global data;
global labels;
global lambda;
nerr = 0;
loss = 0;
for i = start:stop
    [ploss,pnerr] = testOne(data(:,i), labels(i));
    loss = loss + ploss;
    nerr = nerr + pnerr;
end
nerr = nerr/(stop - start + 1);
loss = loss / (stop - start + 1);
cost = loss + 0.5*lambda*anorm;
end

function cost = evaluateEta(start, stop, eta)
global data;
global labels;
global lambda;
global wDivisor;
global aDivisor;
global w;
global a;
global wBias;
global aBias;
global wFraction;
global mu0;

wDivisor = 1;
w = zeros(size(data,1),1);
wBias = 0;
mu0 = 1;
a = zeros(size(data,1),1);
aDivisor = 1;
wFraction = 0;
aBias = 0;

for i = start:stop
    trainOne(data(:,i),labels(i),eta,1);
end
loss = 0;
for i = start:stop
    [ploss,~] = testOne(data(:,i),labels(i));
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