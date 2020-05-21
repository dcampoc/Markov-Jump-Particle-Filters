function [superStateProb,distancePos,mu_i,R_i,distPosFinal] = Projection(currStatePos,averageState,averageDiv,radiusState)

mu_i = [];                     % unitary vector of mu_i
proj = [];
R_i = [];                      % orthogonal component of mu_i
for i =1:1:size(averageState,1)
    mu_i0 = averageDiv(i,:)/norm(averageDiv(i,:));                           % unit vector mu_i
    X0 = currStatePos';                                                      % current state
    P_i0 = averageState(i,:);                                                %
    R_i0 = null(averageDiv(i,:),'r')./vecnorm(null(averageDiv(i,:),'r'));    % FOR HIGHER DIMENSIONS(>2) SELECT ONE OF THE UNITARY VECTORS OR LINEAR COMBINATION OF THEM
%         dot(mu_i0,R_i0);
    n_origin = (dot(X0,mu_i0)/norm(mu_i0)^2).*mu_i0;
    y_origin = (dot(P_i0,R_i0')/norm(R_i0')^2).*R_i0;
    
    projection = n_origin + y_origin';
    proj = [proj; projection];
    mu_i = [mu_i; mu_i0];
    R_i = [R_i R_i0];
end
distancePos = pdist2(X0, proj);
distancePos = distancePos./radiusState;
distPosFinal = distancePos .* (distancePos<1);
inThreshProb = 0.8;                                                           % If the probability of being in a superstate is 'inThreshProb', it is considered to be imposible to be outside the model
outThreshProb = 1 - inThreshProb;
superStateProb = 1 - distancePos;                                             % value 1 means mean and 0 is outside boundary
% superStateProb = softmax(superStateProb');% 
% superStateProb = exp(1-(1./((superStateProb).^2)));
% superStateProb = (exp(superStateProb)./sum(exp(superStateProb)))';
superStateProb = superStateProb .* (distancePos<1);                           % Sum elements
% superStateProb = softmax(superStateProb');
prob = max(superStateProb);                                                   % Calculation of maximum probability
nodesNumber = size(averageState,1);
superStateProb(1,nodesNumber+1) = max([1-(prob+outThreshProb), 0]);           % Probability of empty neuron is 1-probability of most likely neuron
superStateProb = superStateProb/sum(superStateProb);                          % Normalization

% superStateProb = exp(superStateProb)./sum(exp(superStateProb));                        % Normalization

probDistPos = prob;

end