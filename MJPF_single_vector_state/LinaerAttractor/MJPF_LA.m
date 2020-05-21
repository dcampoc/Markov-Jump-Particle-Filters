function estimationAbn = MJPF_LA(averageDiv, covarianceDiv,...
    averageState, covarianceState, radiusState,datanodes, ...
    transitionMat,transMatsTime,curDir, testingData,xydata)

transMatsTime = 0.2;
AS = 1;
plotBool = true;                                                           %   Indicator for plotting results
DivDim = [3 4];
StateDim =  [1  2];

%% testing data
if xydata== true
    dataTest = [testingData.Filtered.xPos, testingData.Filtered.yPos,...
        testingData.Filtered.divxPos, testingData.Filtered.divyPos];
    
else
    dataTest = [testingData.Filtered.S, testingData.Filtered.V,...
        testingData.Filtered.divS, testingData.Filtered.divV];
    
    DataX = testingData.Filtered.xPos;
    DataY = testingData.Filtered.yPos;
    
    DataX = DataX';
    DataY = DataY';
end
%% Definition of the employed data
dataTest = dataTest';
EndPoint = 850;
dataTest = dataTest(:,1:EndPoint);

%% Definition of the parameters
% Transition matrix for the continous-time system.
numState = 2;
A = [eye(numState),zeros(numState);zeros(numState),zeros(numState)];
%   Measurement model.
H = [eye(numState),zeros(numState)];
%   Control input
B = [eye(numState);eye(numState)];
% Variance in the measurements.
r1 = 1e-100;

R = eye(numState)*r1;

nSuperStates = size(averageState,1);                                         %  Number of considered Superstates

%% Initialization Filtering steps.
% Initial guesses for the state mean and covariance.                        % covariance of predetermination noise(covarianza del rumore di predizione)
N = 100;                                                                    % Number of particles
Q = eye(numState*2)*r1*100; % HI                                            % Prediction noise equal for each superstate
weightsPar = zeros(1,N);         %% extra thing                             % Weight of particles (1x100 matrix that contains all zeros)

%% Definition of parameters
statepred = zeros(4,size(dataTest,2),N);                                    % predict state 4D object 1
Ppred = zeros(4,4,size(dataTest,2),N);                                      % predict covariance matrix object 1
stateUpdated = zeros(4,size(dataTest,2),N);
updatedP1 = zeros(4,4,size(dataTest,2),N);
weightscoeff = zeros(N,size(dataTest,2));                                   % weights for each particle(pesi per ogni particle)(WEIGHTS FOR EVERY PARTICLE)

db2 = zeros(N,size(dataTest,2));
innovation1 = zeros(N,size(dataTest,2));
activeNodes = zeros(N,size(dataTest,2));
% velPredictTrans = zeros(N,size(dataTest,2));
abnormMeas = zeros(1,size(dataTest,2));
abnormdb1 = zeros(1,size(dataTest,2));
abnormdb2 = zeros(1,size(dataTest,2));
t = zeros(N,size(dataTest,2));                                              % time for each particle left for the same Superstate(tempo per ogni particella rimasta per lo stesso Superstato)
randLab = zeros(N,1);
nSuperStates = size(transitionMat,2);
firstIt = 1;

%% main loop
%   Counter of trajectories
for i = 1:size(dataTest,2)
    %%   FIRST BLOCK (Initialization of states/update KFs' info)
    %     strfinal = size(dataTest,2);
    %     display([num2str(i) ' out of ' num2str(strfinal)])
    currMeas = dataTest(:,i);
    for n = 1:N                                                             %   For each particle
        if i == firstIt
            %   Adding uncertainty noise for initializing the current state
            currStatePos = awgn(currMeas(1:2,:),100);
            currStateVel = awgn(currMeas(3:4,:),100);
            %   mvnrnd(MU,SIGMA)
            currState = [currStatePos; currStateVel];
            currP1 = eye(4)*r1;                                             % 4x4 matrix contains all zeros except in diagonal elements value specified
            
            stateUpdated(:,i,n) = currState;
            updatedP1(:,:,i,n) = currP1;
            
        else
            %   NEXT MEASUREMENT APPEARS
            % UPDATE
            [stateUpdated(:,i,n),updatedP1(:,:,i,n),~] =...
                kf_update(statepred(:,i-1,n),Ppred(:,:,i-1,n),...
                dataTest(StateDim,i),H,R);%update 1
            updatedP1(:,:,i,n) = updatedP1(:,:,i,n);
            
            %   Association of updated states to variables
            currP1 = updatedP1(:,:,i,n);
            currState = stateUpdated(:,i,n);                                %   updated state of object
            
            currStatePos = currState(StateDim,1);
            currStateVel = currState(DivDim,1);                                %   VELOCITIES ARE NOT CORRECTLY UPDATED!!!!! CHECK THIS!!!!
        end
        
        %%   SECOND BLOCK (Calculation of current superstates)
        
        %   TRANSFORMATION OF STATES INTO SUPERSTATES
        [probDistPos,distancePos,mu_i,R_i,distPosFinal] = Projection(currStatePos,averageState,averageDiv,radiusState);
        %         [probDistPos,distancePos] = nodesProb(currStatePos,averageState,radiusState);
        
        [probDistSort1, propIndSort1] = sort(probDistPos,'descend');        %   Organize in descending order the probabilities of being in each superstate of SOM1 and SOM2
        
        if propIndSort1(1,1) <= nSuperStates
            activeNodes(n,i) = propIndSort1(1,1);
            emptyNBool(i,n) = false;
        else
            probDist = 1./distancePos ; %distPosFinal;
            probDist = probDist/sum(probDist);
            pd = makedist('Multinomial','Probabilities',probDist);          %   Multinomial distribution to pick multiple likely particles
            activeNodes(n,i)  = pd.random(1,1);
            emptyNBool(i,n) = true;
            t(n) = 1;
        end
        
        %   Calculate time in the same discrete state for each particle
        if i == firstIt
            weightscoeff(n,i) = 1/N;                                        %   Weight of particles
        end
        
        
        %% THIRD BLOCK (Abnormal measurements' calculation/particle weighting)
        if i > firstIt
            %   CALCULATION OF ABNORMALITY MEASUREMENTS
            
            if xydata ==1
                %% angular abnormalities
                %                 thetaPred = atan2(statepred(4,i-1), statepred(3,i-1));
                %                 thetaMeas = atan2(currMeas(4,1), currMeas(3,1));
                %                 db2(n,i) = abs(angdiff(thetaPred,thetaMeas));               % restrict the graph between 0 and 1 which is actually in radian
                %% unitary abnormalities
                PredVec = statepred(DivDim,i-1) / norm(statepred(DivDim,i-1));
                MeasVec = currMeas(DivDim,1) / norm(currMeas(DivDim,1));
                db2(n,i) = pdist2(PredVec',MeasVec');
                %% mahalanobis abnormalities
                %                                MeasVec = currMeas(StateDim,1) / norm(currMeas(StateDim,1));
                %                                PredDatanodes = datanodes{1,activeNodes(n, i-1)}(StateDim,:);
                %                                 PredDatanodesTemp = [];
                %                                 for yt = 1:length(PredDatanodes)
                %                                     PredDataNodesNorm = PredDatanodes(yt,:)./norm(PredDatanodes(yt,:));
                %                                     PredDatanodesTemp = [PredDatanodesTemp;PredDataNodesNorm];
                %                                 end
                %                                 db2(n,i) = mahal(MeasVec', PredDatanodesTemp(:,:));
                
            else
                %% angular abnormalities
                %                 thetaPred = atan2(statepred(2,i-1), statepred(1,i-1));
                %                 thetaMeas = atan2(currMeas(2,1), currMeas(1,1));
                %                 db2(n,i) = abs(angdiff(thetaPred,thetaMeas));
                %
                %% unitary abnormalities
                PredVec = statepred(StateDim,i-1)/norm(statepred(StateDim,i-1));
                MeasVec = currMeas(StateDim,1)/ norm(currMeas(StateDim,1));
                db2(n,i) = pdist2(PredVec',MeasVec');
                %%  mahalanobis abnormalities
                %                                 MeasVec = currMeas(StateDim,1) / norm(currMeas(StateDim,1));
                %                                PredDatanodes = datanodes{1,activeNodes(n, i-1)}(StateDim,:);
                %                                 PredDatanodesTemp = [];
                %                                 for yt = 1:length(PredDatanodes)
                %                                     PredDataNodesNorm = PredDatanodes(yt,:)./norm(PredDatanodes(yt,:));
                %                                     PredDatanodesTemp = [PredDatanodesTemp;PredDataNodesNorm];
                %                                 end
                %                                 db2(n,i) = mahal(MeasVec', PredDatanodesTemp(:,:));
            end
            
            
            %             db1 (n, i) = pdist2([averageState(activeNodes(n, i-1),:), ...
            %                 averageDiv(activeNodes(n, i-1), :)], dataTest(:, i-1)');
            db1 (n, i) = pdist2([averageState(activeNodes(n, i-1),:), ...
                averageDiv(activeNodes(n, i-1), :)], dataTest(:, i-1)');
            
            innovation1(n,i) = pdist2(statepred(:,i-1,n)',...
                currMeas(:,1)'); %innovation 1
            
            
            weightsPar(n) = weightscoeff(n,i-1)/(db2(n,i)+1e-10);           % weights are 1/ db1 and db2 of both agents
            
            %% FOURTH BLOCK (wait for all particles, particle resampling)
            if n == N
                % Innovation Measurements
                abnormMeas(i) = min(innovation1(:,i));
                
                % d2 Measurements
                abnormdb2(i) = min(db2(:,i));  % 'mean' is for xy datta
                
                % d1 MeasurementsFIGURE
                abnormdb1(i) = min(db1(:,i));
                
                weightsPar = weightsPar/sum(weightsPar);                    %   Normalize weights in such a way that they all sum 1
                weightscoeff(:,i)= weightsPar';                             %   Assign weights
                
                pd = makedist('Multinomial','Probabilities',weightsPar);    %   Multinomial distribution to pick multiple likely particles
                wRes = pd.random(1,N);                                      %   Take N random numbers from the discrete distribution
                
                for ij = 1:N
                    % REPLACEMENT OF CORRECTED DATA DEPENDING ON
                    % SURVIVING NEURONS
                    temp1 = stateUpdated();
                    temp2 = updatedP1();
                    temp3 = activeNodes();
                    %                     temp4 = t();
                    temp5 = emptyNBool();
                    
                    stateUpdated(:,i,ij) = temp1(:,i,wRes(ij));
                    updatedP1(:,:,i,ij) = temp2(:,:,i,wRes(ij));
                    emptyNBool(i,ij) = temp5(i,wRes(ij));
                    activeNodes(ij,i) = temp3(wRes(ij),i);
                    %                     t(ij) = temp4(wRes(ij));
                    
                    currState = stateUpdated(:,i,ij);
                    currP1 = updatedP1(:,:,i,ij);
                    weightscoeff(ij,i) = 1/N;                               % Weight of particles
                    
                    %% FIFTH BLOCK (prediction of next discrete and continous levels)
                    %   PREDICTION DISCRETE PART (LABELS)
                    transitionCurr = zeros(1,nSuperStates);
                    transitionCurr(1,1:nSuperStates) = transitionMat(activeNodes(ij,i),:);% Pick row of couple superstate(i-1)
                    
                    velPredictProbs = makedist('Multinomial','Probabilities',...          %   Probability of label of two neurons
                        transitionCurr);
                    velPredict(ij,i) = velPredictProbs.random(1,1);                       % Predicted label
%                     velPredictTrans(ij,i) = velPredict(ij,i);
                    %** KALMAN FILTER:
                    % PREDICTION CONTINUOS PART
                    %                     U1 = averageDiv(velPredict(ij,i),1:2)';
                     U1 = (distPosFinal(1,activeNodes(n,i)) * ( R_i(:,activeNodes(n,i)))) + ((1 - distPosFinal(1,activeNodes(n,i))) *  mu_i(activeNodes(n,i),:)');  %mu_i,R_i,distPosFinal
%                     U1 = (distPosFinal(1,velPredictTrans(ij,i)) *  R_i(:,velPredictTrans(ij,i))) + ((1 - distPosFinal(1,velPredictTrans(ij,i))) *  mu_i(velPredictTrans(ij,i),:)');  %mu_i,R_i,distPosFinal
                    
                    [statepred(:,i,ij),Ppred(:,:,i,ij)] =...                                %   Predicition object1
                        kf_predict(currState,currP1, A, Q, B, U1);
                    
                    
                end
                
            end
            
        end
        
        
        %% SIXTH BLOCK (prediction of next discrete and continous levels)
        if i == firstIt
            %   PREDICTION DISCRETE PART (LABELS)
            transitionCurr = zeros(1,nSuperStates);
            transitionCurr(1,1:nSuperStates) =...                           % Pick row of couple superstate(i-1)
                transitionMat(activeNodes(n,i),:);
            
            velPredictProbs = makedist('Multinomial','Probabilities',...    % Probability of label of two neurons
                transitionCurr);
            velPredict(n,i) = velPredictProbs.random(1,1);                  % Predicted label
%             velPredictTrans(n,i) = velPredict(n,i);
            %** KALMAN FILTER:
            %PREDICTION
            U1 = averageDiv(activeNodes(n,i),1:2)';                          % From GNG1
            [statepred(:,i,n),Ppred(:,:,i,n)] =...
                kf_predict(currState,currP1, A, Q, B, U1);                  % Predicition object1
        end
    end
    
    %% SEVENTH BLOCK (Ploting trajectories and abnormality measurements)
    h = figure(1);
    h.Position =[844 63 487 898];
    if xydata == true
        scatter(dataTest(1,1:i),dataTest(2,1:i),'+')
    end
    
    
    if plotBool == true
        if xydata == true
            if i > firstIt
                subplot(4,1,1);
                cla
                hold on
                scatter(dataTest(1,1:i),dataTest(2,1:i),'+')
                axis([-20 30 -5 40])
                title('trajectory')
                
                subplot(4,1,2);
                cla
                plot(abnormdb2(2:i)/0.4495,'-g')   %%% MADE CHANGEEEEEEEEEEEEEEE  /0.4495
                yline(1,'-.b','LineWidth',1); % 0.4441
                title('db2')
                
                subplot(4,1,3);
                cla
                plot(abnormMeas(2:i),'-r')
                %                 yline(0.5417,'-.k','LineWidth',1); % 0.0950
                title('Innovation')
                
                subplot(4,1,4);
                cla
                plot(abnormdb1(2:i),'-r')
                %                 yline(0.4340,'-.k','LineWidth',1); %2.3160
                title('db1')
                
                pause(0.1)
            end
        else
            if i > firstIt
                subplot(6,1,1);
                cla
                quiver3(DataX(1,1:1:i-1),DataY(1,1:1:i-1),dataTest(1,1:1:i-1),diff(DataX(1,1:1:i)),diff(DataY(1,1:1:i)),diff(dataTest(1,1:1:i)),'LineWidth',1.5,'Color','b','AutoScale', 'off')
                axis([-25 30 0 40 -40 30 ])
                title('Steering angle')
                
                subplot(6,1,2);
                cla
                quiver3(DataX(1,1:1:i-1),DataY(1,1:1:i-1),dataTest(2,1:1:i-1),diff(DataX(1,1:1:i)),diff(DataY(1,1:1:i)),diff(dataTest(2,1:1:i)),'LineWidth',1.5,'Color','b','AutoScale', 'off')
                axis([-25 30 0 40 4.5 6.2 ])
                title('Rotor velocity')
                
                subplot(6,1,3);
                cla
                scatter(DataX(1,1:i),DataY(1,1:i),'+')
                title('trajectory')
                subplot(6,1,4);
                cla
                plot(abnormdb2(2:i),'-g')
                axis([0 850 0 1])
                %             yline(5.6701,'-.b','LineWidth',2);
                title('db2')
                
                subplot(6,1,5);
                cla
                plot(abnormMeas(2:i),'-r')
                %             yline(0.8160,'-.k','LineWidth',2);
                title('Innovation')
                
                subplot(6,1,6);
                cla
                plot(abnormdb1(2:i),'-r')
                %             yline(3.6251,'-.k','LineWidth',2);
                title('db1')
                
                pause(0.1)
            end
            
        end
    end
    
    
    estimationAbn.generativeTraj1 = statepred;
    estimationAbn.Dataset = dataTest;
    
    estimationAbn.db2 = abnormdb2;
    estimationAbn.db1 = abnormdb1;
    estimationAbn.innovation = abnormMeas;
    
end


