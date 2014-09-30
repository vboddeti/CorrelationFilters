% compute_psd.m
%
% * Created by Vishnu Naresh Boddeti on 9/30/14.
% * naresh@cmu.edu (http://vishnu.boddeti.net)
% * Copyright 2014 Carnegie Mellon University. All rights reserved.

function compute_psd

global D;
global args;
global data;
global data_freq;

m = args.fft_size(1);
n = args.fft_size(2);
dim = args.dim;
num = size(data,4);
D = zeros(m*n,dim*dim);

if isempty(data_freq)
	batch_size = min(num,args.batch_size);
	num_batches = floor(num/batch_size);
    
    if batch_size >= num
        ind1 = 1;
        ind2 = num;
        index = 1;
        freq = fft2(data,args.fft_size(1),args.fft_size(2))/sqrt(prod(args.fft_size));
        for p = 1:dim
            for q = 1:dim
                tmp = sum((freq(:,:,p,ind1:ind2)).*conj(freq(:,:,q,ind1:ind2)),4);
                D(:,index) = D(:,index) + tmp(:);
                index = index + 1;
            end
        end
    else
        for i = 1:num_batches
            ind1 = (i-1)*batch_size + 1;
            ind2 = i*batch_size;
            freq = fft2(data(:,:,:,ind1:ind2),args.fft_size(1),args.fft_size(2))/sqrt(prod(args.fft_size));
            index = 1;
            for p = 1:dim
                for q = 1:dim
                    tmp = sum((freq(:,:,p,:)).*conj(freq(:,:,q,:)),4);
                    D(:,index) = D(:,index) + tmp(:);
                    index = index + 1;
                end
            end
        end
        
        ind1 = i*batch_size + 1;
        ind2 = num;
        freq = fft2(data(:,:,:,ind1:ind2),args.fft_size(1),args.fft_size(2))/sqrt(prod(args.fft_size));
        index = 1;
        for p = 1:dim
            for q = 1:dim
                tmp = sum((freq(:,:,p,:)).*conj(freq(:,:,q,:)),4);
                D(:,index) = D(:,index) + tmp(:);
                index = index + 1;
            end
        end 
    end
else
	batch_size = min(num,args.batch_size);
	num_batches = floor(num/batch_size);
	
    if batch_size >= num
        ind1 = 1;
        ind2 = num;
        index = 1;
        for p = 1:dim
            for q = 1:dim
                tmp = sum((data_freq(:,:,p,ind1:ind2)).*conj(data_freq(:,:,q,ind1:ind2)),4);
                D(:,index) = D(:,index) + tmp(:);
                index = index + 1;
            end
        end
    else
        for i = 1:num_batches
            ind1 = (i-1)*batch_size + 1;
            ind2 = i*batch_size;
            index = 1;
            for p = 1:dim
                for q = 1:dim
                    tmp = sum((data_freq(:,:,p,ind1:ind2)).*conj(data_freq(:,:,q,ind1:ind2)),4);
                    D(:,index) = D(:,index) + tmp(:);
                    index = index + 1;
                end
            end
        end
        
        ind1 = i*batch_size + 1;
        ind2 = num;
        index = 1;
        for p = 1:dim
            for q = 1:dim
                tmp = sum((data_freq(:,:,p,ind1:ind2)).*conj(data_freq(:,:,q,ind1:ind2)),4);
                D(:,index) = D(:,index) + tmp(:);
                index = index + 1;
            end
        end
    end
end

D = D/num;
