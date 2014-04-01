function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n = size(X,2);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

tempo = zeros(1,m);

for i = 1:m
    first_half = -y(i) * log(sigmoid(theta'*X(i,:)'));
    second_half= (1 - y(i)) * log(1 - sigmoid(theta'*X(i,:)'));
    total = first_half - second_half;
    tempo(i) = total * (1/m);
end

J = sum(tempo);

for ii = 1:n
    tempo = zeros(1,m);
    for jj = 1:m
        hypothesis = sigmoid(theta'*X(jj,:)');
        diff = hypothesis - y(jj);
        prod = diff * X(jj,ii);
        tempo(jj) = prod;
    end
    grad(ii) = (1/m)*sum(tempo);
end
% =============================================================

end
