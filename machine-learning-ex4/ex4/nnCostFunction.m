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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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

% Add the bias node to the input
a1 = [ones(m, 1) X];

% Pass through the hidden layer
z2 = Theta1 * a1';
a2 = sigmoid(z2);

% Add the bias node to the output
a2 = [ones(1, columns(a2)); a2];

% Pass to the output layer
z3 = Theta2 * a2;
a3 = sigmoid(z3);

% Transpose the outputs
pred = a3';

% One hot encode the ground truth
labels = zeros(m,num_labels);
for i = 1:size(y,1),
  labels(i, y(i)) = 1;
end

% Calculate the cost
J = (1/m) * sum(sum(-labels .* log(pred) - (1 - labels) .* log(1 - pred), 2));

% Add the regularization term
% Remove the first column, which is the bias term
reg_term = lambda/(2*m) * (sum(sum(Theta1(:,2:columns(Theta1)).^2)) + ...
			   sum(sum(Theta2(:,2:columns(Theta2)).^2)));
J += reg_term;

% Perform backprop
Delta3 = zeros(1, num_labels);
Delta2 = zeros(hidden_layer_size, hidden_layer_size + 1);
Del2 = zeros(size(Theta2_grad));
Del1 = zeros(size(Theta1_grad));

for t = 1:m,
  a1 = [1 X(t,:)];
  z2 = Theta1 * a1';
  a2 = sigmoid(z2);
  a2 = [1; a2];
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  pred = a3';

  Delta3 = pred - labels(t,:);
  Delta2 = Delta3 * Theta2;
  Delta2 = Delta2(:,2:columns(Delta2));
  Delta2 = Delta2 .* sigmoidGradient(z2)';

  Del2 += (a2 * Delta3)';
  Del1 += Delta2' * a1;
end

Theta1_grad = (1/m) .* Del1;
Theta2_grad = (1/m) .* Del2;

% Regularize the gradients
Theta1_grad(:,2:columns(Theta1_grad)) += lambda/m * Theta1(:,2:columns(Theta1));
Theta2_grad(:,2:columns(Theta2_grad)) += lambda/m * Theta2(:,2:columns(Theta2));
  
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
