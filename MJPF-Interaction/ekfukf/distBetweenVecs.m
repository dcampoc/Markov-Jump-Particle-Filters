%	Function that neasures the distances between a state and each element of a vocabulary
%	Inputs: 
%	State: vector n x 1, where n represents the number of states. 
%	average: matrix m x n, where m represents the number of letters in the vocabulary.
%	radious: maximum distance from which the model is not valid 
%	[beta, alpha]: weights for position and velocities respectively

%	Outputs:
%	distProb: 1 x m vectr containing the probability of the observed state to be part of each letter
%	dist2Voc: m x 1 vector containing the distance from the state to each letter 

function distance = distBetweenVecs(state1, state2)
currS1 = state1(1,1); currS2 = state1(2,1);
protM1 = state2(1,1); protM2 = state2(2,1);                 
distance = sqrt((protM1 - currS1).^2 + (protM2 - currS2).^2);