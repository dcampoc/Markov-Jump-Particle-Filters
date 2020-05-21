%% Fourth  Algorithm: (Abnormality detection)
close all
clear
clc
curDir = pwd;
set(0,'defaultfigurecolor',[1 1 1])
load('VocabularyN.mat')

datatTrainigBool = false;                                                   % True for training data and false for testing
xydata = true;                                                              % True for odometry and false for control(steering and velocity) use just for training purposes
Testing = 1;                                                                % cases 1 = uturn , cases 2 = obstacle avoidance  and case 3 = steering angle and velocity
%% Data for testing
if datatTrainigBool == true
    load('PMDatafile.mat')
else
    
    if Testing == 1
        load('UturnDatafile.mat') %% U Turn position
%          load('PMDatafile.mat')
    elseif Testing == 2
        load('OADatafile.mat')%% Obstacle Avoidance position
        
    elseif Testing == 3
        load('ESDatafile.mat')%% Emergency Stop position
        
    elseif Testing == 4
        load('ESDatafile.mat') %% Emerggency Stop Control Data
        
    elseif Testing == 5
        load('UturnDatafile.mat')%% U Turn Control Data
        
    elseif Testing == 6
        load('OADatafile.mat')%% Obstacle Avoidance Control Data
        
    end
end

inputData = structSyncData;

 
%% Mean and Covariance (training data)

averageState = netP.nodesMean(:,[1,2]);                                        %   Mean neurons of position data
averageDiv = netP.nodesMean(:,[3,4]);

%   Covariance of position data
for i =1:netP.N
    temp1 = netP.nodesCov{i};
    split_temp12 = temp1([1,2],[1,2]);
    split_temp34 = temp1([3,4],[3,4]);
    covarianceState{i} = split_temp12;                                         %   Covariance of position data
    covarianceDiv{i} = split_temp34 ;                                          %   Covariance of velocity data


end

radiusState = netP.nodesRadAccept;                                               %   Acceptance neuron radius of position data

% transMatsTime = netP.TimeMats;                                        %   Transition time of Som of couples (labels)
transitionMat = netP.transitionMat;                                   %   Transition time of Som of couples (labels)
transMatsTime = 0.2;
cd(curDir)
figure;scatter(netP.data(:,1),netP.data(:,2)); hold on; scatter(structSyncData.Filtered.xPos,structSyncData.Filtered.yPos,'r')
%% MJPF application
estimationAbn = MJPF_LA(averageDiv, covarianceDiv,...
    averageState, covarianceState, radiusState,netP.datanodes,...
    transitionMat,transMatsTime,curDir, inputData,xydata);                   % true for xydata and false for SV data


figure;
plot(smooth(estimationAbn.db2))

if datatTrainigBool == true
    
    if xyTrain == true
        namefile = 'TrainAbnSig.mat';
        save(namefile,'estimationAbn');
        
    else
        namefile = 'SVTrainAbnSig.mat';
        save(namefile,'estimationAbn');
    end
else
    if Testing == 1
        namefile = 'PosAbnSig.mat';
        save(namefile,'estimationAbn');
        
    elseif Testing == 2
        namefile = 'OAAbnSig.mat';
        save(namefile,'estimationAbn');
    elseif Testing == 3
        namefile = 'ESPMAbnSig.mat';
        save(namefile,'estimationAbn');
    elseif Testing == 4
        namefile = 'ESSVAbnSig.mat';
        save(namefile,'estimationAbn');
        
    elseif Testing == 5
        namefile = 'SVAbnSig.mat';
        save(namefile,'estimationAbn');
    elseif Testing == 6
        namefile = 'SVOAAbnSig.mat';
        save(namefile,'estimationAbn');
    end
end




