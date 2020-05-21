function[superStateProb, index, distFinal1,distances2] = neuron(currState,averageMat,radius, alpha, beta)
alphaVal = alpha/2;
betaVal = beta/6;
inThreshProb = 0.8;                                                        %   If the probability of being in a superstate is 'inThreshProb', it is considered to be imposible to be outside the model  
outThreshProb = 1 - inThreshProb;

somSize = size(averageMat,1);
distances = distance8(currState,averageMat);
distances2 = distances';
vecWeights = [betaVal; betaVal; alphaVal; alphaVal; betaVal ;betaVal;...
    betaVal ; betaVal];
distFinal = (distances2./radius).*repmat(vecWeights,1,size(distances2,2));
distFinal = sum(distFinal);
distFinal1 = distFinal';
superStateProb(1,1:size(distances,1)) = 1 - distFinal;
superStateProb = superStateProb .* (superStateProb>0);                      % Sum elements
[prob, index] = max(superStateProb);                                        % Index of maximum probability
superStateProb(1,somSize+1) = max([1-(prob+outThreshProb), 0]);                                       %   Probability of empty neuron is 1-probability of most likely neuron
superStateProb = superStateProb/sum(superStateProb);                        % Normalization
end