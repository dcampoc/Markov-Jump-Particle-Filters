%   SECOND ALGORITHM (vocabulary)
%   Application of SOM over trajectories to create vocabulary of dynamics
%   Input: ['Training' num2str(numExp)]: contiunus state space of tracker and attractor
%   And the number of each trajectory ['numberData' num2str(numExp)
%   Output: ['vocabularyGen' num2str(numData)'.mat'] and M.mat
%   Notes: this good should be run two times, first time 'object = 1' and
%   Second time 'object = 2'.

clc
clear
close all
curDir = pwd;
set(0,'defaultfigurecolor',[1 1 1])
addpath('./somtoolbox');
object = 1;

%% LOAD DATA 
% cd(dirMat)
load('pos1.mat')                                                            % Continuos state space of follower
load('pos2.mat')                                                            % Continuos state space of attractor
data = [pos1 pos2];                                                         % 8D[Xt Yt Xt_dot Yt_dot Xa Ya Xa_dot Ya_dot] vector of follower and attractor (input of SOM)
load('numberData.mat')                                                      % Number of data for each trajectory
data2 = [pos2 pos1];                                                        % 8D[Xa Ya Xa_dot Ya_dot Xt Yt Xt_dot Yt_dot ] vector of attractor and follower (input of SOM)

%% TRAINING VOCABULARY
% Weights definition for position (beta), velocity (alpha)
alpha = 0.85;
beta = 0.15;

somAlphaBeta = [alpha, beta];
%
alphaVal = alpha/2;                                                         %   Distribute the importance of alpha in the coresponding parameters
betaVal = beta/6;                                                           %   Distribute the importance of beta in the coresponding parameters
if object == 1
    %   Object 1: Favors follower's velocity
    [M, containerID, dataCode2, sizeSOM, averageN,...
        covarianceN, containerNumbData, usedNeurons, containerData,...
        distNeighborsStat, colorsMats, sizeSOM1, sizeSOM2] = somclustering(alphaVal,...
        betaVal, data);                                                      % SOM and radius of certainty boundary
        radiusPos = distNeighborsStat.Pos.radius;
    
    %% Plot approximated data by SOM in xDot and ydot
    colSOM = colorsMats.SOM(usedNeurons,:);
    avNmat = cell2mat(averageN);
    h2 = figure;
    hold on;
    s2 = scatter(avNmat(:,1), avNmat(:,2),50,'k', 'filled');       %   Draws neuron with z=v_x
    scatter(data(:,1),data(:,2),...                         %   Data on the plane
        4,colorsMats.Data, 'filled');
    xlab = xlabel('$x$','interpreter','latex');
    ylab = ylabel('$y$','interpreter','latex');
    zlab = zlabel('$\dot{x}$','interpreter','latex');
    xlab.FontSize = 22;
    ylab.FontSize = 22;
    zlab.FontSize = 22;
    grid minor
%     leg2 = legend(s2,'Prototypes projected onto $\dot{x}$');
else
    %   Object 2: Favors attractor's velocity
    [M, containerID, dataCode2, sizeSOM, averageN,...
        covarianceN, containerNumbData, usedNeurons, containerData,...
        distNeighborsStat, colorsMats,  sizeSOM1, sizeSOM2] = somclustering(alphaVal, betaVal, data2);
    % Calculation of neurons borders
    radiusPos = distNeighborsStat.Pos.radius;

    %% Plot approximated data by SOM in xDot and ydot
    %   xDot components
    colSOM = colorsMats.SOM(usedNeurons,:);
    avNmat = cell2mat(averageN);
    h2 = figure;
    hold on;
    s2 = scatter(avNmat(:,1), avNmat(:,2),80,'k', 'filled');       %   Draws neuron with z=v_x
    scatter(data2(:,1),data2(:,2),...                         %   Data on the plane
        4,colorsMats.Data, 'filled');
    xlab = xlabel('$x$','interpreter','latex');
    ylab = ylabel('$y$','interpreter','latex');
    zlab = zlabel('$\dot{x}$','interpreter','latex');
    xlab.FontSize = 22;
    ylab.FontSize = 22;
    zlab.FontSize = 22;
    grid minor
%     leg2 = legend(s2,'Prototypes projected onto $\dot{x}$');

end

%% Greate overall vocabulary
if object == 2
%     cd(dirMat)
    load('vocabularyGen1.mat')                                                  %   Load information from clusters related to SOM that favors follower's velocity
%     cd(curDir)
    dataCode1 = vocabularyGen.dataCode;
    [label,transMat] = transitionProbability(dataCode1,dataCode2,...            %   Occurence of all possible combinations of coupled superstates (neurons)
        sizeSOM,numberData); %   Transition matrix_superstate
    transMatsTimeFinal = ...                   %   Transition matrix of coupled superstates (neurons)
        transitionTim(label,dataCode1,dataCode2, sizeSOM,transMat,numberData);
    
else
    transMat = [];
    transMatsTimeFinal = [];

end
% Save outputs
    vocabularyGen.prototypes = M;                                               %   Prototypes of states vectors
    vocabularyGen.transitionMatrix = transMat;                                  %   Transition matrix of model
    vocabularyGen.containerID = containerID;                                    %   Cell with indexes of data in neurons
    vocabularyGen.dataCode = dataCode2;                                          %   Array of crossed neurons
    vocabularyGen.sizeSom = sizeSOM;                                            %   Number of neurons in the SOM
    vocabularyGen.averageN =  averageN;                                         %   Average of neuron
    vocabularyGen.covarianceN = covarianceN;                                    %   Covariance of neuron
    vocabularyGen.countDataN = containerNumbData;                               %   For each couple of superstates is percentage of having it after a time instant
    vocabularyGen.usedNeuronsLab = usedNeurons;                                 %   Labels of employed neurons
    vocabularyGen.containerData = containerData;                                %   Array of data inside each neuron
    vocabularyGen.transMatsTime = transMatsTimeFinal;                           %   Matrix with histograms of time
    vocabularyGen.distNeighStatistics = distNeighborsStat;                      %   Statistics related to the closest distances between created neurons and their standard deviation
    vocabularyGen.somAlphaBeta = somAlphaBeta;                                  %   Parameters used to weight the SOM
    
    if(object == 2)
        vocabularyGen.label = label;
    end
    filename = ['vocabularyGen' num2str(object) '.mat'];
    save (filename,'vocabularyGen')
