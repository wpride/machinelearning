function [errs, C_ind, sig_ind] = testcsigma()

test_values = [.01, .03, .1, .3, 1, 3, 10, 30];

load('ex6data3.mat')

errors = zeros(1, numel(test_values)^2);
Cs = zeros(1, numel(test_values)^2);
sigmas = zeros(1, numel(test_values)^2);

iterator = 0;

for idx = 1:numel(test_values)
    for idx2 = 1:numel(test_values)
        
        iterator = iterator + 1;
        
        C = test_values(idx);
        sigma = test_values(idx2);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        
        predictions = svmPredict(model, Xval);
        
        error = mean(double(predictions ~= yval));
        
        errors(1,iterator) = error;
        Cs(1,iterator) = C;
        sigmas(1, iterator) = sigma;
        
        % visualizeBoundary(X, y, model);
        % pause;
    end
end

index = find(errors == min(errors))
C_ret = Cs(index)
sig_ret = sigmas(index)

errs = errors;
C_ind = Cs;
sig_ind = sigmas;
% Try different SVM Parameters here
% [C, sigma] = dataset3Params(X, y, Xval, yval);

% Train the SVM
% model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
% visualizeBoundary(X, y, model);

end

