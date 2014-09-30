function [w, bias] = linear_training(training, X, y)
%   LINEAR_TRAINING
%   Linear training algorithms for real and complex data.
%
%   [W, BIAS] = LINEAR_TRAINING(TRAINING, X, Y)
%   Trains a linear classifier/regressor with data matrix X
%   (samples-by-features), and labels/regression targets Y.
%
%   The algorithm is specified as a field of the TRAINING struct:
%
%   training.type = 'svm':     Support Vector Machine (requires liblinear)
%   training.type = 'svr':     Support Vector Regression (req. liblinear)
%   training.type = 'ridge':   Ridge Regression
%
%   Parameters:
%
%   training.regularization: The regularization parameter, which is C for
%     SVM/SVR, and Lambda for Ridge Regression.
%
%   training.epsilon: Parameter for the epsilon-insensitive loss of SVR.
%
%   training.bias_term: Value of the constant feature that is added as a
%     bias term (typically 1), or 0 for no bias.
%
%   training.complex: If true, the SVR will solve an extended problem to
%     support complex data (Ridge Regression already does it by default).
%
%   Joao F. Henriques, 2013

	
	%if needed, add a (regularized) bias term, as a constant feature
	if training.bias_term,
		X(:,end+1) = training.bias_term;
	end
	
	switch training.type
	case 'ridge',
		%ridge regression
		
		w = (X' * X + training.regularization * eye(size(X,2))) \ (X' * y);
		
		w = conj(w);  %only has an effect on complex data
		
		
	case {'svm', 'svr'},
		%liblinear for large-scale linear SVM/SVR
		
		if training.complex,  %extended SVR for complex regression
			X = [real(X), imag(X); imag(X), -real(X)];
			y = [real(y); imag(y)];
		end
		
		%compose string to call liblinear
		if strcmp(training.type, 'svm'),  %SVM
			str = '-s 1';  %dual L2-L2 SVM
% 			str = '-s 2';  %primal L2-L2 SVM
		else  %SVR
			str = ['-s 11 -p ' num2str(training.epsilon)];  %primal L2-L2 SVR
% 			str = ['-s 12 -p ' num2str(training.epsilon)];  %dual L2-L2 SVR
		end
		
		str = [str ' -c ' num2str(training.regularization) ' -q'];
		
		%call it
		s = train(y, sparse(X), str);
		w = s.w(:);
		
		if training.complex,  %unpack result of complex regression
			w = w(1:end/2) + 1i * w(end/2+1:end);
		end
		
	otherwise
		error(['Unknown training type: ' training.type '.'])
	end
	
	%unpack bias result
	if training.bias_term,
		bias = w(end) * training.bias_term;
		w(end) = [];
	else
		bias = 0;
	end

end
