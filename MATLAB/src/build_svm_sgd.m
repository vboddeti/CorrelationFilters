% build_svm_sgd.m

function out = build_svm_sgd

global args;
global data;
global labels;

global w;
global wBias;
global lambda;
global wDivisor;
global eta0;
global t;

[m,n,dim,num] = size(data);

lambda = 1/(args.C*num);
num_epochs = args.num_epochs;
data = reshape(data,[m*n*dim,num]);
ind = randperm(num);
data = data(:,ind);
labels = labels(ind);
eta0 = determineEta0(1,num);

t = 0;
wBias = 0;
wDivisor = 1;
w = zeros(m*n*dim,1);

for epoch = 1:num_epochs
    ind = randperm(num);
    data = data(:,ind);
    labels = labels(ind);
    train(1,num, eta0);
    [cost,loss,nerr] = test(1, num);
    disp([epoch cost loss nerr]);
end

out.filt = reshape(w,[m,n,dim]);
out.b = wBias;

data = reshape(data,[m,n,dim,num]);
[labels,ind] = sort(labels,'descend');
data = data(:,:,:,ind);
end

function [] = renorm
global w;
global wDivisor;
if (wDivisor ~= 1)
    w = w/wDivisor;
    wDivisor = 1;
end
end

function w_norm = wnorm
global w;
global wBias;
global wDivisor;
w_norm = (w'*w)/(wDivisor*wDivisor);
w_norm = w_norm + wBias*wBias;
end

function [ploss,pnerr] = testOne(x,y)
global w;
global wBias;
global wDivisor;
s = (w'*x)/wDivisor + wBias;
ploss = max(0,1-s*y);
pnerr = (s*y <= 0);
end

function [] = trainOne(x,y,eta)
global w;
global wBias;
global wDivisor;
global lambda;
s = (w'*x)/wDivisor + wBias;
wDivisor = wDivisor/(1-eta*lambda);
if (wDivisor > 1e5)
    renorm;
end
etab = eta * 0.01;
wBias = wBias*(1 - etab*lambda);
z = s*y;
if z < 1
    w = w + eta*y*wDivisor*x;
    wBias = wBias + etab*y;
end
end

function [] = train(start, stop, eta0)
global t;
global data;
global labels;
global lambda;
for i = start:stop
    eta = eta0/(1+lambda*eta0*t);
    trainOne(data(:,i), labels(i), eta);
    t=t+1;
end
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
cost = loss + 0.5*lambda*wnorm;
end

function cost = evaluateEta(start, stop, eta)
global w;
global data;
global labels;
global lambda;
global wBias;
global wDivisor;

wBias = 0;
wDivisor = 1;
w = zeros(size(data,1),1);

for i = start:stop
    trainOne(data(:,i),labels(i),eta);
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