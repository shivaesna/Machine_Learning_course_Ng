function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h_theta = X * theta;
product = h_theta - y;
theta_n = theta([2:size(theta,1)],:);
J= 1/(2*m)*sum(product.^2) + lambda/(2*m) * sum(theta_n.^2);

grad0 = 1/m * sum( (h_theta -y) .* X)';
grad1 = grad0 + lambda/m * theta;
grad = [grad0(1,1);
    grad1([2:size(grad1,1)],:)];




% =========================================================================

grad = grad(:);

end
