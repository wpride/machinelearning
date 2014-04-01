    function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
thetas = size(theta, 1);
features = size(X,2)

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    summer_0 = 0;
    summer_1 = 0;
    
    coefficient = (1/m);
    full_coefficient = alpha * coefficient;
    
    for j=1:m
        theta_0_new = theta(1) + theta(2)*X(j,2) - y(j);
        theta_1_new = (theta(1) + theta(2)*X(j,2) - y(j)) * X(j,2);
        summer_0 = summer_0 + theta_0_new;
        summer_1 = summer_1 + theta_1_new;
    end
    
    theta(1) = theta(1) - full_coefficient * summer_0;
    theta(2) = theta(2) - full_coefficient * summer_1;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    
    

end

end
