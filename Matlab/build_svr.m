% build_svr.m
%
% * Citation: This function implements the correlation filter design proposed in the publications.
% * João F. Henriques, João Carreira, Rui Caseiro and Jorge Batista, "Beyond Hard Negative Mining:
% * Efficient Detector Learning via Block-Circulant Decomposition", ICCV, 2013.
% 
% The implementation is a slight modification of the original implementation.
% * Created by Vishnu Naresh Boddeti on 9/30/14.
% * naresh@cmu.edu (http://vishnu.boddeti.net)

function out = build_svr

global args;
global labels;
global data_freq;

fft_scale_factor = sqrt(prod(args.fft_size));

pos_labels = gaussian_shaped_labels(args.target_magnitude, args.target_sigma, args.img_size(1:2));
neg_labels = -args.target_magnitude * ones(args.img_size(1:2));

pos_labels = pos_labels - mean2(pos_labels);
neg_labels = neg_labels - mean2(neg_labels);

pos_labels = fft2(pos_labels, args.fft_size(1), args.fft_size(2))/fft_scale_factor;
neg_labels = fft2(neg_labels, args.fft_size(1), args.fft_size(2))/fft_scale_factor;

num = length(labels);
num_pos_samples = sum(labels==1);

weights = zeros([args.fft_size args.dim]);

training.type = args.filt_type;
training.regularization = args.alpha;  %SVR-C, obtained by cross-validation on a log. scale
training.epsilon = args.C;
training.complex = true;
training.bias_term = 0;

if ~args.parallel_flag,
    %circulant decomposition (non-parallel code).
    
    y = zeros(num,1);  %sample labels (for a fixed frequency)
    tic
    for r = 1:args.fft_size(1)
        for c = 1:args.fft_size(2)
            %fill vector of sample labels for this frequency
            y(:) = neg_labels(r,c);
            y(1:num_pos_samples) = pos_labels(r,c);
            
            %train classifier for this frequency
            weights(r,c,:) = linear_training(training, double(permute(data_freq(r,c,:,:), [4, 3, 1, 2])), y);
        end
        
        progress(r, args.fft_size(1));
    end
    toc
    
else
    %circulant decomposition (parallel code).
    %to use "parfor", we have to deal with a number of issues.
    
    %first, split data into chunks, stored in a cell array, to avoid the
    %2GB data-transfer limit of "parfor" (bug fixed in MATLAB2013a). (*)
    disp('Chunking data to avoid MATLAB''s data transfer limit...')
    samples_chunks = cell(args.fft_size(1),1);
    for r = 1:args.fft_size(1),
        samples_chunks{r} = data_freq(r,:,:,:);
    end
    clear samples;
    tic
    parfor r = 1:args.fft_size(1),
        %normally we'd set "weights(r,c,:)" as in the non-parallel code,
        %but "parfor" doesn't like complicated indexing. so the inner loop
        %will build just one row, and only then we store it in "weights".
        row_weights = zeros([args.fft_size(2) args.dim]);
        
        y = zeros(num,1);  %sample labels (for a fixed frequency)
        
        for c = 1:args.fft_size(2),
            %fill vector of sample labels for this frequency
            y(:) = neg_labels(r,c);
            y(1:num_pos_samples) = pos_labels(r,c);
            
            row_weights(c,:) = linear_training(training, double(permute(samples_chunks{r}(1,c,:,:), [4, 3, 1, 2])), y);
            
            % 			%with MATLAB2013a or newer, you can comment out the chunking
            % 			%code (*), and use this to train with "samples" directly:
            % 			row_weights(c,:) = linear_training(training, double(permute(samples(r,c,:,:), [4, 3, 1, 2])), y);
        end
        
        weights(r,:,:) = row_weights;  %store results for this row of weights
    end
    toc
end

out.b = 0;
out.filt_freq = h_prox(weights,args);
tmp = ifft2(out.filt_freq,'symmetric')*fft_scale_factor;
out.filt = tmp(1:args.img_size(1),1:args.img_size(2),:);
out.args = args;