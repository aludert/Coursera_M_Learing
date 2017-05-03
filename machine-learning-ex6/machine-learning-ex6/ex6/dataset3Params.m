function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

error = 10e+10000000;
C_out = 10e+10000000;
sigma_out = 10e+10000000;

C_vect = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vect = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

for ic = 1:length(C_vect)
    C_temp = C_vect(ic);
    
    for isigma = 1:length(sigma_vect)
        sigma_temp = sigma_vect(isigma);
        
        model= svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
        predictions = svmPredict(model, Xval);
        error_temp = mean(double(predictions ~= yval));
        
        if error_temp < error
            C_out = C_temp;
            sigma_out = sigma_temp;
            error = error_temp;
        end
        
    end
    
end

C = C_out;
sigma = sigma_out;


% =========================================================================

end
