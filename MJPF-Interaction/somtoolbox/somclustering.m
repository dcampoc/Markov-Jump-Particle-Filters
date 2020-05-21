function[M,containerID, dataCode, sizeSOM, averageN,...
    covarianceN, containerNumbData, usedNeurons, containerData,...
    distNeighborsStat, colorsMats, m1, m2] = somclustering(alpha, beta, dataOri)
    data = dataOri;
%% SOM takes input training set
[a,~] = size(data);                                                         %   Size a is number of input vectors
m1 = 10;%10
m2 = 11;%11
sizeSOM = m1*m2;

%% SOM algorithm
%initialization of SOM
smI = som_lininit(data, 'msize', [m1 m2] , 'lattice', 'hexa', ...
    'shape', 'sheet');
smR = som_batchtrainDistance(smI, data, alpha, beta,...                     %   Coarse training
    'radius', [5 .5], 'trainlen', 50, 'neigh', 'gaussian');
sm = som_batchtrainDistance(smR, data, alpha, beta,...                       %   Fine training with same radius
    'radius', [0.5 0.1], 'trainlen',  50, 'neigh', 'gaussian');

M = sm.codebook;                                                           %   M are the prototypes(neurons) of genererated neurons

colorsSOM = som_coloring(sm);                                              %   Colors that group the different prototypes
h = gcf;
som_cplane('hexa', [m1 m2], colorsSOM);                                    %   Draw of similarity between generated prototypes
h.Position = [2507 189 756 696];
%   divide in 8 components neuron to compute the  distance
M1 = M(:,1);
M2 = M(:,2);
M3 = M(:,3);
M4 = M(:,4);
M5 = M(:,5);
M6 = M(:,6);
M7 = M(:,7);
M8 = M(:,8);

%% Calculation of membership of data to each generated prototype
containerNumbData = zeros(m1*m2,1);                                         %   Counter of data in neuron
containerID = cell(1,m1*m2);                                                %   Indexes of data in neuron
containerData = cell(1,m1*m2);                                              %   Vector of data in neuron
dataCode = zeros(a,1);                                                      %   Coded data (neuron where it was encoded)
dataCodeDist = zeros(a,1);                                                  %   Distance to the proposed prototype
colorsData = [];
%   Distances between data and prototypes
for i = 1: a                                                                %   For all input vectors
    I1 = data(i,1);
    I2 = data(i,2);
    I3 = data(i,3);
    I4 = data(i,4);
    I5 = data(i,5);
    I6 = data(i,6);
    I7 = data(i,7);
    I8 = data(i,8);
    Y = sqrt(beta*((M1 - I1).^2) + beta*((M2 - I2).^2) +...                 %   Weighted distance between all centroids and a particular data point (entire system vector)
        beta*((M5 - I5).^2) + beta*((M6 - I6).^2) +...
        beta*((M7 - I7).^2) + beta*((M8 - I8).^2) +...
        alpha*((M3 - I3).^2) + alpha*((M4 - I4).^2));
    [dist,c] = min(Y);                                                      %   Find minimum neuron to the state in question
    dataCodeDist(i) = dist;                                                 %   Distance to the closes neuron
    dataCode(i) = c;                                                        %   Closest neuron to the data in question
    containerNumbData(c,1) = containerNumbData(c) + 1;                                                  %   Add one to counter of input data in c unit
    containerID{c}(containerNumbData(c),:) = i;                                          %   Indexes of data asociated to each neuron
    containerData{c}(containerNumbData(c),:) = dataOri(i,:);                                %   Add input vector to neuron
    colorsData = [colorsData; colorsSOM(c,:)];
end
colorsMats.Data = colorsData;
colorsMats.SOM = colorsSOM;
distNeighborsStat.Total.radius = ones(8,sizeSOM)*1e+100;
usedNeurons = [];
for i = 1:sizeSOM
        usedNeurons = [usedNeurons; i];                                    %   Label of used neurons
        averageN{i,1} = mean(containerData{i});                            %   Mean of node
        covarianceN{i,1} = cov(containerData{i});                          %   Covariance of node
        % check positive definite
        eig_A = eig(covarianceN{i,1});                                     %   Eigenvalues and eigenvectors
        flag = 0;
        for d = 1:rank(covarianceN{i,1})
            if eig_A(d) <= 0
                flag = 1;
            end
        end
        if flag == 1
            covarianceN{i,1} = abs(covarianceN{i,1});
        else
            disp('the matrix is positive definite')
        end          
        covariance2 = diag(covarianceN{i,1}); 
        radiusTot = 3*sqrt(covariance2); 
        distNeighborsStat.Total.radius(:,i) = radiusTot;        
        radiusPos = [radiusTot(1:2,1); radiusTot(5:6,1)];
        distNeighborsStat.Pos.radius(:,i) = radiusTot;
end

%% Calculation of additional functions
drawNumberOfNeurons(containerNumbData,m1, m2);                              %   Number of vector in nodes
unifiedMatrix(sm);                                                          %   Display unified matrix
end