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

C_try = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
Sigma_try = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];



for i = 1:size(C_try,2)
    for j = 1:size(Sigma_try,2)
        model = svmTrain(X, y, C_try(1,i), @(x1, x2)gaussianKernel(x1, x2, Sigma_try(1,j)));
        predictions = svmPredict(model, Xval);
        Error_Eval(i,j) = mean(double(predictions ~= yval));
    end
end

[M,I] = min(Error_Eval);
[N,J] = min(M);
%Incecis = [I(J) J];
C_index = I(J);
Sigma_Index = J;

C = C_try(1,C_index);
sigma = Sigma_try(1,Sigma_Index);



% =========================================================================

end
