function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(X, 2);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

tempo = zeros(1,m);

for i = 1:m
    first = -y(i) * log(sigmoid(theta'*X(i,:)'));
    second= (1 - y(i)) * log(1 - sigmoid(theta'*X(i,:)'));
    total = first - second;
    tempo(i) = total * (1/m);
end

reg_cost = zeros(1,n)

for i = 2:n
    reg_cost(i) = theta(i)^2;
end

full_reg_cost = sum(reg_cost) * (lambda/(2*m));

J = sum(tempo) + full_reg_cost;

for ii = 1:n
    tempo = zeros(1,m);
    for jj = 1:m
        hypothesis = sigmoid(theta'*X(jj,:)');
        diff = hypothesis - y(jj);
        prod = diff * X(jj,ii);
        tempo(jj) = prod;
    end
    if ii > 1
        grad(ii) = (1/m)*sum(tempo) + (lambda/m)*theta(ii);
    else
        grad(ii) = (1/m)*sum(tempo);
    end
end




% =============================================================

end
