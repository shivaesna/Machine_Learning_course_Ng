function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z=theta'*X';
h_theta= sigmoid(z);
sum=0;
sum1=0;
for i=1:m
    sum1=sum1+(-y(i)*log(h_theta(i))-(1-y(i))*log(1-h_theta(i)));
end

sum2=0;
for j=2:size(theta)
    sum2=sum2+theta(j,1)^2;
end

J=1/m*sum1+lambda/(2*m)*sum2;


sum_theta=zeros(size(grad));
for j=1:size(grad)
  for i=1:m
    sum_theta(j,1)=sum_theta(j,1)+(h_theta(i)-y(i))*X(i,j);
  end
end

grad=1/m*sum_theta;

for j=2:size(theta)
    grad(j,1)=grad(j,1)+lambda/m*theta(j,1);
end




% =============================================================

end
