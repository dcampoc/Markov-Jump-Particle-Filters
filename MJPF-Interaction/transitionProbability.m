function [label,transMat] = transitionProbability(dataCode1,dataCode2,side,numberData)
%% Find transitionMat
eTot = side*side;                                                           %   Total number of events
label = zeros(eTot,4);                                                      %   Combination of all possible events
for i = 1:side
    label(side*(i-1)+1:side*(i),1) = i;                                     %   Initialize the first column
    label(side*(i-1)+1:side*(i),2) = 1:side;                                %   Initialize the second column
end
% Assign the identifiers for the events
label(:,3) = 1:eTot;
dataCode = [dataCode1 dataCode2];
label2 = [];
sum2 = 0;
%In?
for i = 1:size(numberData,1)
    for b = 1 + sum2:numberData(i) + sum2
        label2 = [label2 ; dataCode(b,:)];
    end
    sum2 = sum2 + numberData(i);
end
%Ou?
label2 = unique(label2, 'rows');                                           % remove repetitions (and set in sorted order)
%IN?
newlabel = 1:size(label2);
newlabel = newlabel';
%Ou?
label2 = horzcat(label2, newlabel);
for j = 1:size(label2,1)
    label(((label2(j,1)-1)*side + label2(j,2)),4) = label2(j,3);
end
transMat = zeros(size(label2,1));
sum2 = 0;
for i = 1:size(numberData,1)
    for b = sum2+1:numberData(i) + sum2 - 1                                 % -1 to distinguish between the end to a trajectory with tha start to another one(because we dont have exactly the transition between them)
        coderow = (dataCode(b,1)-1)*side + dataCode(b,2);
        codecol = (dataCode(b+1,1)-1)*side + dataCode(b+1,2);
        rowMat = label(coderow,4);                                             %   Row of transition matrix is first node
        colMat = label(codecol,4);
        transMat(rowMat,colMat) = transMat(rowMat,colMat) + 1;                 %   Increase occurrence of transition
    end
    sum2 = sum2 + numberData(i);
end
%% Normalize relative frequency
normVect = sum(transMat,2) + (sum(transMat,2) == 0);
transMat = transMat./repmat(normVect,1,size(transMat,1));
%% visualize the transition matrix
figure
imagesc(transMat);
% colorbar;
% colormap jet
title('Transtion matrix for superstates')
end