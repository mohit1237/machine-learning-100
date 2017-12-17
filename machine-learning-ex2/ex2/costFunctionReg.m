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

p=1/m;
z=X*theta;
h=1./(1.+exp(-z));
theta1=theta(2:end,:);
theta2=theta(1,:);
J=(p*((y'*(-1.*log(h))).-((1.-y)'*(log(1.-h)))))+((lambda*p)/2)*sum(theta1.^2);
zampa=p*((h-y)'*X)';
grad(1,:)=zampa(1,:);
grad(2:end,:)=zampa(2:end,:)+(lambda*p)*theta1;




% =============================================================

end
