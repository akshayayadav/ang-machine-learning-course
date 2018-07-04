function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
hx=X*theta;

% You need to return the following variables correctly 
J = 0;
theta_feat=theta([2:length(theta)],1);
J=sum((hx-y).^2)/(2*m) + ((lambda/(2*m)) * (theta_feat' * theta_feat));

grad = zeros(size(theta));
num_iters=length(theta);
grad(1,1) = (sum((hx-y)'*X(:,1)))/m;
for iter = 2:num_iters
  grad(iter,1) = ((sum((hx-y)'*X(:,iter)))/m) + ((lambda/m)*theta(iter,1));


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
