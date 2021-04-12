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

lowest_error = -1;
possible_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
best_C = possible_vals(1);
best_sigma = possible_vals(1);

for i = possible_vals,
    for j = possible_vals,

        model = svmTrain(X, y, i, @(x1, x2) gaussianKernel(x1, x2, j));
        predictions = svmPredict(model, Xval);
        pred_error = mean(double(predictions ~= yval));

        if lowest_error == -1 || pred_error < lowest_error,
            lowest_error = pred_error;
            best_C = i;
            best_sigma = j;
        end

    end
end

C = best_C;
sigma = best_sigma;

% =========================================================================

end
