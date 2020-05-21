function EstimationAbn = MJPFilter(averageN2, covarianceN2, usedNeuronsLab2,...
    averageN1, covarianceN1, usedNeuronsLab1,distNeighStatistics1,...
    distNeighStatistics2,transitionMat,transMatsTime,...
    somAlphaBeta,codeBook,rMid,deltaR,vecWeights,track)
%%  Add package for special computations
addpath('./ekfukf')
%% Definition of the employed data
load('numberDataTest.mat')                                                  %   Number of data for each trajectory of test
load('pos1Test.mat')                                                        %   Data test object 1
load('pos2Test.mat')                                                        %   Data test object 2
data = [pos1Test pos2Test];
dataTest = data';
startInd = sum(numberDataTest(1:track-1,1)) + 1;
endInd = sum(numberDataTest(1:track-1,1)) + numberDataTest(track);
alpha = somAlphaBeta(1,1);
beta  = somAlphaBeta(1,2);
%% Definition of the parameters
% Transition matrix for the continous-time system.
dtModel = 1;                                                               % Time stamp
A = [1 0  0  0;
    0  1  0  0;
    0  0  0  0;
    0  0  0  0];                                                           % Dynamic model
%   Measurement model.
H = [1 0 0 0
    0 1 0 0];
%   Control input
B = [  dtModel 0;
    0 dtModel;
    1 0;
    0 1];
% Variance in the measurements.
r1 = 1e-4;
R = diag([r1 r1]);
somSize1 = size(averageN1,1);
somSize2 = size(averageN2,1);
% number of neurons in each som
somSizeLab = size(transitionMat,1);                                         % number of couples in training
%% Initialization Filtering steps.
Pinit = diag([r1 r1 r1 r1]);                                               % Guess for P
% Initial guesses for the state mean and covariance.
N = 100;                                                                    % Number of particles
Q = 100*eye(4)*r1;                                                          % prediction noise equal for each superstate
% Radius of superstates
radius1 = distNeighStatistics1.Total.radius;                                % radius for each superstate som 1
radius2 = distNeighStatistics2.Total.radius;                                % radius for each superstate som 2
% Mean and covariance are equal to the Vocabulary
averageMat1 = ones(somSize1,8)*1e+100 ;                                      %average matrix of som 1
averageMat2 = ones(somSize1,8)*1e+100 ;                                       %average matrix of som 2
meanU1 = ones(somSize1,2)*1e+100;                                           %   U vector with velocity for som 1
meanU2 = ones(somSize1,2)*1e+100;                                           %   U vector with velocity for som 2
covarianceMat1 = ones(8,8,somSize1)*1e+100;                                 %covariance matrix for superstates of som 1
covarianceMat2 = ones(8,8,somSize1)*1e+100;                                 %covariance matrix for superstates of som 2
for i = 1:size(usedNeuronsLab1,1)                                           %   Set control input variable for each neuron average for U
    ii = usedNeuronsLab1(i,1);
    covarianceMat1(:,:,ii) = covarianceN1{ii,1};
    meanU1(ii,:) = averageN1{ii,1}(1,3:4);
    averageMat1(ii,:) = averageN1{ii,1};
end
for i = 1:size(usedNeuronsLab2,1)                                           %   Set control input variable for each neuron average for U
    ii = usedNeuronsLab2(i,1);
    covarianceMat2(:,:,ii) = covarianceN2{ii,1};
    meanU2(ii,:) = averageN2{ii,1}(1,3:4);
    averageMat2(ii,:) = averageN2{ii,1};
end
w = zeros(1,N);                                                            % pesi di particle
h2 = figure(2);
h2.Position = [441 43 491 922];
hold on
%% Definition of parameters
subMatx2 = zeros(4);
subMatx3 = zeros(4);
statepred1 = zeros(4,size(dataTest,2),N);                                   % predict state 4D object 1
Ppred1 = zeros(4,4,size(dataTest,2),N);                                     % predict covariance matrix object 1
statepred2 = zeros(4,size(dataTest,2),N);                                  % predict state 4D object 2
Ppred2 = zeros(4,4,size(dataTest,2),N);                                    % predict covariance matrix object 2
stateUpdated1 = zeros(4,size(dataTest,2),N);
updatedP1 = zeros(4,4,size(dataTest,2),N);
stateUpdated2 = zeros(4,size(dataTest,2),N);                                % stato aggiornato
updatedP2 = zeros(4,4,size(dataTest,2),N);                                         % P aggiornato
weightscoeff = zeros(N,size(dataTest,2));                                   % pesi per ogni particle
s1 = zeros(N,size(dataTest,2));                                             % superstato del 1
s2 = zeros(N,size(dataTest,2));                                             % superstato del 2
d11 = zeros(N,size(dataTest,2));
d12 = zeros(N,size(dataTest,2));
db21 = zeros(N,size(dataTest,2));
db22 = zeros(N,size(dataTest,2));
innovation1 = zeros(N,size(dataTest,2));
innovation2 = zeros(N,size(dataTest,2));
activeLabels = zeros(N,size(dataTest,2));
abnormMeas = cell(2,endInd);
abnormdb2 = cell(2,endInd);
abnormdb1 = cell(2,endInd);
t = zeros(N,size(dataTest,2));                                             % tempo per ogni particella rimasta per lo stesso Superstato
a = track;
currStateHist = [];
temp2 = zeros(2,N);
temp = zeros(2,N);
%% main loop
for i = startInd:endInd
    %% 1) particle propagation
    %   Counter of particles for each data sample
    for n = 1:N                                                        % For each particle
        if i == startInd
            subMatx1 = [eye(2)*r1/5 zeros(2); zeros(2) eye(2)*r1/5];
            subMatx4 = [eye(2)*r1/5 zeros(2); zeros(2) eye(2)*r1/5];
            PinitFull = [subMatx1 subMatx2;subMatx3 subMatx4];
            currState = mvnrnd(dataTest(:,i),PinitFull)';                 % find initial states of 2 objects
            currStateHist = [currStateHist currState];
            currState1 = currState(1:4,:);                              %current state object 1
            currState2 = currState(5:8,:);                              %current state object2
            currP1 = Pinit;
            currP2 = Pinit;
            stateUpdated1(:,i,n) = currState1;
            stateUpdated2(:,i,n) = currState2;
            updatedP2(:,:,i,n) = currP1;
            updatedP1(:,:,i,n) = currP1;
        else
            %   NEXT MEASUREMENT APPEARS
            % UPDATE
            [stateUpdated1(:,i,n),updatedP1(:,:,i,n),~] =...
                kf_update(statepred1(:,i-1,n),Ppred1(:,:,i-1,n),...
                dataTest(1:2,i),H,R);%update 1
            [stateUpdated2(:,i,n),updatedP2(:,:,i,n),~] =...
                kf_update(statepred2(:,i-1,n),Ppred2(:,:,i-1,n),...
                dataTest(5:6,i),H,R);%update 2
            updatedP1(:,:,i,n) = positivedefinite(updatedP1(:,:,i,n));
            updatedP2(:,:,i,n) = positivedefinite(updatedP2(:,:,i,n));
            
            %   Association of updated states to variables
            currP1 = updatedP1(:,:,i,n);
            currP2 = updatedP2(:,:,i,n);
            PinitFull = [currP1 subMatx2;subMatx3 currP2];
            currState1 = stateUpdated1(:,i,n);                          % updated state object 1
            currState2 = stateUpdated2(:,i,n);                          % current state object2
            currState = [currState1; currState2];
            currState = mvnrnd(currState,PinitFull/100)';               % find initial states of 2 objects
            currState1 = currState(1:4,:);                              %current state object 1
            currState2 = currState(5:8,:);                              %current state object2
        end
        
        %   TRANSFORMATION OF STATES INTO SUPERSTATES
        [probDist1,~, distance1, distComplete1] =...
            neuron(currState',averageMat1,radius1, alpha, beta);        %   Find probability of being in each superstate of SOM1
        [probDist2,~, distance2,distComplete2] =...
            neuron([currState2;currState1]',...
            averageMat2,radius2, alpha, beta);                          %   Find probability of being in each superstate of SOM2
        
        [probDistSort1, propIndSort1] = sort(probDist1,'descend');      %   Organize in descending order the probabilities of being in each superstate of SOM1 and SOM2
        [probDistSort2, propIndSort2] = sort(probDist2,'descend');
        i1 = 1;
        i2 = 1;
        
        %%  Loop for transforming continuous state to superstate
        while(true)
            emptyNBool(i,n) = false;
            if i1 > somSize1 || i2 > somSize2
                %   All combinations are tried and no label is identified
                emptyNBool(i,n) = true;
                minDistLabel = 0;
                break
            end
            ind1Curr = propIndSort1(i1);
            ind2Curr = propIndSort2(i2);
            if ind1Curr > somSize1 || ind2Curr > somSize2
                %   The closest label to the current state is the empty neuron
                emptyNBool(i,n) = true;
                minDistLabel = 0;
                break
            end
            codeEval = codeBook((ind1Curr-1)*somSize1+ind2Curr,4) ~= 0;
            s1Curr = probDistSort1(i1);
            s2Curr = probDistSort2(i2);
            if codeEval == true
                s1(n,i) = ind1Curr;
                s2(n,i) = ind2Curr;
                minDistLabel = codeBook((s1(n,i)-1)*...
                    somSize1+s2(n,i),4); % look if this couple has been observed
                break
                
            else
                sTot  = [s1Curr + probDistSort2(i2+1),...
                    s2Curr + probDistSort1(i1+1)];
                [~,sIndMax] = max(sTot);
                if sIndMax == 1
                    i2 = i2+1;
                else
                    i1 = i1+1;
                end
            end
            if s1Curr == 0 || s2Curr == 0
                %   The state is outside from at least 1 neuron
                emptyNBool(i,n) = true;
                minDistLabel = 0;
                break
            end
        end
        %   find minimum sum
        if minDistLabel == 0
            sumDistance = distance1 + distance2;
            [~, minDistLabel] = min(sumDistance);
            
        end
        activeLabels(n,i) = minDistLabel;
        
        if i > startInd
            if (activeLabels(n,i-1)== activeLabels(n,i)) && (emptyNBool(i-1,n) == false) && (emptyNBool(i,n) == false)
                t(n) = t(n) + 1;                                        % If same pair add 1
            else
                t(n) = 1;                                               % Else rinizialize by 1
            end
        else
            t(n) = 1;                                                   %   Time spend in a label is initialized as 1
            weightscoeff(n,i) = 1/N;                                    %   Weight of particles
        end
        indActiveLab = find(codeBook(:,4) == activeLabels(n,i));
        neurSomInd1 = codeBook(indActiveLab,1);
        neurSomInd2 = codeBook(indActiveLab,2);
        distSom1{i,n}  = distComplete1(:,neurSomInd1);
        distSom2{i,n}  = distComplete2(:,neurSomInd2,1);
        
        if i > startInd
            %   CALCULATION OF ABNORMALITY MEASUREMENTS
            db21(n,i) = bhattacharyyadistance(statepred1(1:2,i-1,n)',...        % measure bhattacharrya distance between p(xk/zk-1) and p(zk/xk) object 1
                dataTest(1:2,i)',Ppred1(1:2,1:2,i-1,n), R);
            db22(n,i) = bhattacharyyadistance(statepred2(1:2,i-1,n)',...       % measure bhattacharrya distance between p(xk/zk-1) and p(zk/xk) object 2
                dataTest(5:6,i)',Ppred2(1:2,1:2,i-1,n), R);
            d11(n,i) = bhattacharyyadistance(statepred1(:,i-1,n)',...
                averageMat1(neurSomInd1,1:4),Ppred1(:,:,i-1,n),...            %   measure bhattacharrya distance between p(xk/zk-1) and p(xk/sk) object 1
                positivedefinite(covarianceMat1(1:4,1:4,neurSomInd1)));
            d12(n,i) = bhattacharyyadistance(statepred2(:,i-1,n)',...
                averageMat2(neurSomInd2,1:4),Ppred2(:,:,i-1,n),...     % measure bhattacharrya distance between p(xk/zk-1) and p(xk/sk) object 2
                positivedefinite(covarianceMat2(1:4,1:4,neurSomInd2)));
            
            % CALCULATION OF INNOVATIONS
            innovation1(n,i) = distBetweenVecs(statepred1(:,i-1,n), dataTest(1:2,i)); %innovation 1
            innovation2(n,i) = distBetweenVecs(statepred2(:,i-1,n), dataTest(5:6,i)); %innovation 2
            
            w(n) = weightscoeff(n,i-1)/(d11(n,i)+d12(n,i)+db21(n,i)+db22(n,i))+1e-10;   %weights are 1/ db1 and db2 of both agents
            
            % RESAMPLING OF PARTICLES
            if n == N
                % Innovation Measurements
                abnormMeas{1,a} = [abnormMeas{1,a}, min(innovation1(:,i))];
                abnormMeas{2,a} = [abnormMeas{2,a}, min(innovation2(:,i))];
                
                % d2 Measurements
                abnormdb2{1,a} = [abnormdb2{1,a}, min(db21(:,i))];
                abnormdb2{2,a} = [abnormdb2{2,a}, min(db22(:,i))];
                
                % d1 Measurements
                abnormdb1{1,a} = [abnormdb1{1,a}, min(d11(:,i))];
                abnormdb1{2,a} = [abnormdb1{2,a}, min(d12(:,i))];
                
                w = w/sum(w);                                           %   Normalize weights in such a way that they all sum 1
                weightscoeff(:,i)= w';                                  %   Assign weights
                
                pd = makedist('Multinomial','Probabilities',w);         %   Multinomial distribution to pick multiple likely particles
                wRes = pd.random(1,N);                                  %   Take N random numbers from the
                check = 1;
                for ij = 1:N
                    % REPLACEMENT OF CORRECTED DATA DEPENDING ON
                    % SURVIVING NEURONS
                    stateUpdated1(:,i,ij) = stateUpdated1(:,i,wRes(ij));
                    updatedP1(:,:,i,ij) = updatedP1(:,:,i,wRes(ij));
                    stateUpdated2(:,i,ij) = stateUpdated2(:,i,wRes(ij));
                    updatedP2(:,:,i,ij) = updatedP2(:,:,i,wRes(ij));
                    %   Association of updated states to variables
                    currP1 = updatedP1(:,:,i,ij);
                    currP2 = updatedP2(:,:,i,ij);
                    PinitFull = [currP1 subMatx2;subMatx3 currP2];
                    currState1 = stateUpdated1(:,i,ij);                          % updated state object 1
                    currState2 = stateUpdated2(:,i,ij);                          % current state object2
                    currState = [currState1; currState2];
                    currState = mvnrnd(currState,PinitFull)';                   % find initial states of 2 objects
                    currState1 = currState(1:4,:);                              %current state object 1
                    currState2 = currState(5:8,:);                              %current state object2
                    emptyNBool(i,ij) = emptyNBool(i,wRes(ij));
                    activeLabels(ij,i) = activeLabels(wRes(ij),i);
                    t(ij) = t(wRes(ij));
                    weightscoeff(ij,i) = 1/N;                           %   Weight of particles
                    
                    distSom1{i,ij} = distSom1{i,wRes(ij)};          %   Replace the distances to the closer couple neurons (in the resampling step)
                    distSom2{i,ij} = distSom2{i,wRes(ij)};
                    % PREDICTION DISCRETE PART
                    transitionCurr = zeros(1,somSizeLab+1);
                    transitionCurr(1,1:somSizeLab) = transitionMat(activeLabels(ij,i),:);% Pick row of couple superstate(i-1)
                    if(t(ij) <= size(transMatsTime,2))                       %   If time staying in a single region is normal
                        matTr = transitionCurr(1,1:somSizeLab);
                        matTime = transMatsTime{1,t(ij)}(activeLabels(ij,i),:)+1e-10;
                        transitionCurr(1,1:somSizeLab) = (matTr).*matTime;    %multiply by time interval probability
                    else                                                    %   If time staying in a single region is abnormal
                        transitionCurr(1,1:somSizeLab) = ones(1,somSizeLab)*1e-10;
                    end
                    sum3 = sum(transitionCurr);
                    if(sum3~=0)
                        transitionCurr = transitionCurr/sum3;               %   Normalize probabilities
                    end
                    %   Defining probabilities of going in an empty label
                    %%%%%%%%%%%%%%%%%%%%%%%%%%
                    distRef1 = sum((distSom1{i,ij}./rMid.SOM1).*vecWeights);
                    distRef2 = sum((distSom2{i,ij}./rMid.SOM2).*vecWeights);
                    distTot = mean([distRef1 distRef2]);
                    if distTot < 1
                        transitionCurr(1,end) = 0;
                    else
                        probEmpty1 = abs(distSom1{i,ij}-rMid.SOM1)./deltaR.SOM1;
                        probEmpty2 = abs(distSom2{i,ij}-rMid.SOM2)./deltaR.SOM2;
                        probEmpty1 = sum(probEmpty1.*vecWeights);
                        probEmpty2 = sum(probEmpty2.*vecWeights);
                        probEmptyTot = mean([probEmpty1 probEmpty2]);
                        transitionCurr(1,end) = min([1 probEmptyTot]);
                    end
                    transitionCurr(1,1:end-1) =  transitionCurr(1,1:end-1)/...
                        sum(transitionCurr(1,1:end-1))*(1 -...
                        transitionCurr(1,end));
                    labPredictProbs = makedist('Multinomial','Probabilities',...    %   probability of label of two neurons
                        transitionCurr);
                    labPredict(ij,i) = labPredictProbs.random(1,1);      % Predicted label
                    
                    if labPredict(ij,i) < somSizeLab + 1
                        indPred = find(codeBook(:,4)==labPredict(ij,i));
                        som1Pred(ij,i) = codeBook(indPred,1);
                        som2Pred(ij,i) = codeBook(indPred,2);
                    end
                    %** KALMAN FILTER:
                    % PREDICTION CONTINUOS PART
                    if(labPredict(ij,i) == (somSizeLab + 1))                         %   If predisted label is empty
                        U1 = [0 0]';                                                %   We do not have U
                        U2 = [0 0]';
                    else
                        U1 = meanU1(som1Pred(ij,i),:)';                 %   From som1
                        U2 = meanU2(som2Pred(ij,i),:)';                 %   From som2
                    end
                    [statepred1(:,i,ij),Ppred1(:,:,i,ij)] =...
                        kf_predict(currState1,currP1, A, Q, B, U1);     %   predicition object1
                    [statepred2(:,i,ij),Ppred2(:,:,i,ij)] =...
                        kf_predict(currState2,currP2, A, Q, B, U2);     %   predicition object2
                    temp(:,n) = statepred1(1:2,i,n);
                    temp2(:,n) = statepred2(1:2,i,n);
                end
            end
        end
        if i == startInd
            %% Prediction of discrete part
            transitionCurr = zeros(1,somSizeLab+1);
            transitionCurr(1,1:somSizeLab) = transitionMat(activeLabels(n,i),:);% Pick row of couple superstate(i-1)
            if(t(n) <= size(transMatsTime,2))                       %   If time staying in a single region is normal
                matTr = transitionCurr(1,1:somSizeLab);
                matTime = transMatsTime{1,t(n)}(activeLabels(n,i),:)+1e-10;
                transitionCurr(1,1:somSizeLab) = (matTr).*matTime;    %multiply by time interval probability
            else                                                    %   If time staying in a single region is abnormal
                transitionCurr(1,1:somSizeLab) = ones(1,somSizeLab)*1e-10;
            end
            sum3 = sum(transitionCurr);
            if(sum3~=0)
                transitionCurr = transitionCurr/sum3;               %   Normalize probabilities
            end
            %   Defining probabilities of going in an empty label
            distRef1 = sum((distSom1{i,n}./rMid.SOM1).*vecWeights);
            distRef2 = sum((distSom2{i,n}./rMid.SOM2).*vecWeights);
            distTot = mean([distRef1 distRef2]);
            
            if distTot < 1
                transitionCurr(1,end) = 0;
            else
                probEmpty1 = abs(distSom1{i,n}-rMid.SOM1)./deltaR.SOM1;
                probEmpty2 = abs(distSom2{i,n}-rMid.SOM2)./deltaR.SOM2;
                
                probEmpty1 = sum(probEmpty1.*vecWeights);
                probEmpty2 = sum(probEmpty2.*vecWeights);
                
                probEmptyTot = mean([probEmpty1 probEmpty2]);
                transitionCurr(1,end) = min([1 probEmptyTot]);
            end
            transitionCurr(1,1:end-1) =  transitionCurr(1,1:end-1)/...
                sum(transitionCurr(1,1:end-1))*(1 -...
                transitionCurr(1,end));
            
            labPredictProbs = makedist('Multinomial','Probabilities',...    %   probability of label of two neurons
                transitionCurr);
            labPredict(n,i) = labPredictProbs.random(1,1);                  % Predicted label
            
            if labPredict(n,i) < somSizeLab + 1
                indPred = find(codeBook(:,4)==labPredict(n,i));
                som1Pred(n,i) = codeBook(indPred,1);
                som2Pred(n,i) = codeBook(indPred,2);
            end
            %** KALMAN FILTER:
            %PREDICTION
            if(labPredict(n,i) == (somSizeLab + 1))                         %   If predisted label is empty
                U1 = [0 0]';                                                %   We do not have U
                U2 = [0 0]';
            else
                U1 = meanU1(som1Pred(n,i),:)';                                   %   From som1
                U2 = meanU2(som2Pred(n,i),:)';                                   %   From som2
            end
            
            [statepred1(:,i,n),Ppred1(:,:,i,n)] =...
                kf_predict(currState1,currP1, A, Q, B, U1);                 %predicition object1
            
            [statepred2(:,i,n),Ppred2(:,:,i,n)] =...
                kf_predict(currState2,currP2, A, Q, B, U2);                 %predicition object2
            temp(:,n) = statepred1(1:2,i,n);
            temp2(:,n) = statepred2(1:2,i,n);
        end
    end
    EstimationAbn.abnormMeas1 = abnormMeas{1,track};
    EstimationAbn.abnormMeas2 = abnormMeas{2,track};
    % db1
    EstimationAbn.abnormdb11 = abnormdb1{1,track};
    EstimationAbn.abnormdb12 = abnormdb1{2,track};
    % db2
    EstimationAbn.abnormdb21 = abnormdb2{1,track};
    EstimationAbn.abnormdb22 = abnormdb2{2,track};
    subplot(3,1,1);
    axis([-20 20 -20 20])
    %
    if track == 28
        scatter(0,-2,1000,'MarkerFaceColor','y','MarkerEdgeColor','k',...
            'MarkerFaceAlpha',.1,'MarkerEdgeAlpha',0.25,'LineWidth',1);
        ff = text(-1.8,-2.8,'Obstacle');
        ff.FontSize = 10;
        ff.FontWeight = 'bold';
    end
    hold on
    scatter(dataTest(1,i),dataTest(2,i),10,'r','filled')
    scatter(dataTest(5,i),dataTest(6,i),10,'b','filled')
    scatter(temp2(1,:),temp2(2,:),w/max(w)*10,'filled','g');%particles
    scatter(temp(1,:),temp(2,:),w/max(w)*10,'filled','k');%particles
    subplot(3,1,2);
    cla
    plot(abnormMeas{1,a} ,'-r','LineWidth',2)
    subplot(3,1,3);
    cla
    plot(abnormMeas{2,a} ,'-b','LineWidth',2)
    pause(0.05)
    
end


end
