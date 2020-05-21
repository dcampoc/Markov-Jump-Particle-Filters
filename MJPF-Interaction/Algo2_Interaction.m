% MARKOV JUMP PARITCLE FILTER AND ABNORMALITY DETECTION.
clc
clear
close all
track = 28; %(track = 28)abnormal , (track = 18)normal
%% LOAD INPUT AND DEFINE INDEX OF TRAINING AND TESTING                                                          
%   Vocabulary information of entire system that favors object 1 (follower) dynamics 
load(['vocabularyGen',num2str(1),'.mat'])                                   
averageN1 = vocabularyGen.averageN;                                         %   Mean neurons of som1 (Follower)
covarianceN1 = vocabularyGen.covarianceN;                                   %   Covariance of som1 (Follower)
usedNeuronsLab1 = vocabularyGen.usedNeuronsLab;                             %   Used neurons (Follower)
distNeighStatistics1 = vocabularyGen.distNeighStatistics;                   %   Acceptance neuron radius (Follower) 
%   Vocabulary information of entire system that favors object 2 (attractor) dynamics 
load(['vocabularyGen',num2str(2),'.mat'])                                   
averageN2 = vocabularyGen.averageN;                                         %   Mean neurons of som2 (Att)
covarianceN2 = vocabularyGen.covarianceN;                                   %   Covariance of som2 (Att)
usedNeuronsLab2 = vocabularyGen.usedNeuronsLab;                             %   Used neurons (Att)
distNeighStatistics2 = vocabularyGen.distNeighStatistics;                   %   Acceptance neuron radius (Att) 
%   Info of coupled objects (Follower + Attractor)
transMatsTime = vocabularyGen.transMatsTime;                                %   Transition time of Som of couples (labels)
label = vocabularyGen.label;                                                %   Matrix containing all possible couples (fisrt two columns), a counter of all of them (third column) and counter of observed labels  
somAlphaBeta = vocabularyGen.somAlphaBeta;                                  %   Information about weights of alpha and beta for prioritizing the dynamics of an object
transitionMat = vocabularyGen.transitionMatrix;                             %   Transition matrix of couples (labels)
% Threshold 
radius1 = distNeighStatistics1.Total.radius;                                %   radius for each superstate som 1
radius2 = distNeighStatistics2.Total.radius;                                %   radius for each superstate som 2

ind = find(label(:,4)~=0);
som1Act = label(ind,1);
som2Act = label(ind,2);

alpha = somAlphaBeta(1,1);
beta = somAlphaBeta(1,2);

alphaVal = alpha/2;
betaVal = beta/6;

vecWeights = [betaVal; betaVal; alphaVal; alphaVal; betaVal ;betaVal;...
    betaVal ; betaVal];
radiusSom1 = radius1(:,som1Act);
radiusSom2 = radius2(:,som2Act);

rMid.SOM1 = median(radiusSom1,2);
rMid.SOM2 = median(radiusSom2,2);
deltaR.SOM1 = 3*std(radiusSom1,[],2);                                       %   Limit where the prediction goes definetly outside the model being outside 
deltaR.SOM2 = 3*std(radiusSom2,[],2);
%% Satellite Animal model
EstimationAbn = MJPFilter(averageN2, covarianceN2, usedNeuronsLab2,...
    averageN1, covarianceN1, usedNeuronsLab1,distNeighStatistics1,...
    distNeighStatistics2, transitionMat,transMatsTime,...
    somAlphaBeta,label,rMid,deltaR,vecWeights,track);