function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                 


% Setup some useful variables
m = size(X, 1);


class_labels=1:num_labels;
% You need to return the following variables correctly 
J = 0;

for num_datapt = 1:m
  y_lab = class_labels == y(num_datapt,1);
  
  X_i=[1 X(num_datapt,:)];
  lay2_out = sigmoid(X_i*Theta1');
  lay2_out = [1 lay2_out];
  hx = sigmoid(lay2_out*Theta2');
  
  J_i = sum((-y_lab .* log(hx)) - ((1-y_lab).*log(1-hx)));
  
  J=J+(J_i/m);
end

Theta1_temp=Theta1(:,2:(input_layer_size + 1));
Theta2_temp=Theta2(:,2:(hidden_layer_size + 1));
nn_params_temp=[Theta1_temp(:) ; Theta2_temp(:)];

params_count=size(nn_params_temp,1);
params_sum=0;

for param_i = 1:params_count
  params_sum=params_sum+(nn_params_temp(param_i,1)**2);
end
params_sum=params_sum*(lambda/(2*m));
J=J+params_sum;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for num_datapt = 1:m
  y_lab = class_labels == y(num_datapt,1);
  y_lab=y_lab';
  
  X_i=[1 X(num_datapt,:)];
  lay2_out = sigmoid(X_i*Theta1');
  lay2_out = [1 lay2_out];
  hx = sigmoid(lay2_out*Theta2');
  hx=hx';
  
  delta_3 = hx-y_lab;
  sigmoid_grad=sigmoidGradient(Theta1*X_i');
  delta_2 = (Theta2'*delta_3).*sigmoid_grad;
  delta_2 = delta_2(2:end);
  Theta2_grad = Theta2_grad + (delta_3*lay2_out);
  Theta1_grad = Theta1_grad + (delta_2*X_i');
  end

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
