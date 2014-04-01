function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    tempo = [];
    result = [];
    theta_temp = [];
    
    %for all the thetas
    for t =1:thetas
        %for all the examples
        for examples = 1:m
            tempo(examples) = ((theta' * X(examples, :)') - y(examples)) *X(examples,t);
        end
        
        result(t) = sum(tempo);
        tempo = 0;
        
    end
    
    for c= 1:thetas
       theta_temp(c) = theta(c) - alpha * (1/m) * result(c); 
    end
    
    for j = 1:thetas
        theta(j) = theta_temp(j);
    end
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end
end
