function transMatsTimeFinal1 =...
    transitionTim(label,dataCode,dataCode1,sizeSOM,transMat,numberData)
%find matrix
sizeDictionary = max(label(:,4));
row = 0;
list = [];
for a = 1:sizeDictionary
    for b = 1:sizeDictionary
        if(a~=b && transMat(a,b)~=0)                                        %   If nodes are different
            list = [list; [a b]];                                           %   Organization of all observed transitions
            row = row + 1;
            transMatTime{row} = [a b];
        elseif(a==b && transMat(a,b)~=0)                                    %   If nodes are the same
            list = [list; [a b]];                                           %   Organization of all observed transitions
            row = row + 1;
            transMatTime{row} = [a b];
        end
    end
end
% We need to identify all the events along any individual trajectory
% and store them into the corresponding "seq"
previous = 0;
previous2 = 0;
sum2 = 0;
for i = 1:size(numberData,1)%for each path
    for j = 1:numberData(i)-1
        a = label((dataCode(j+sum2)-1)*sizeSOM + dataCode1(j+sum2),4);
        b = label((dataCode(j+sum2+1)-1)*sizeSOM + dataCode1(j+sum2+1),4);
        [row,~] = find((list(:,1) == a).*...                     %   Identification of transition
            (list(:,2) == b));
        if(a ~= b)
            previous2 = 0;                                                      %   Superstate is changed
            if previous == 0                                                    %   First chenge
                transMatTime{row} = [transMatTime{row} j+sum2-1];                    %   Append the time where the transition finished
                previous = j+sum2;                                                   %   Assign current time as previous
            else
                elapsed = j+sum2 - previous;                                         %   Calculation of time elapsed
                transMatTime{row} = [transMatTime{row} elapsed-1];                %   Time elapsed in the initial zones before making the proposed transition
                previous = j+sum2;
            end
        else  %if I am in same neuron and compute from the initial time how much I've stayed
            if previous2 == 0
                transMatTime{row} = [transMatTime{row} 1];                      %   Add 1 to time (elapse time in the same zone)
                previous2 = 2;                                                  %   Next time it will be 2 if it continues to stay at the same zone
            else
                transMatTime{row} = [transMatTime{row} previous2];
                previous2 = previous2+1;
            end
        end
    end
    sum2=sum2+numberData(i);
end
maxDuration = 0;                                                            %   Maximum time elapsed in a zone
for i = 1:size(transMatTime,2)
    if(max(transMatTime{i}(3:end))> maxDuration)
        maxDuration = max(transMatTime{i}(3:end));
    end
end

nbins = 0:1:maxDuration;
for i = 1:size(transMatTime,2)
    [countOccurTime{i,1},timeIntervElaps] = hist(transMatTime{i}(3:end),nbins);       %   In first 2 elements number of
end

countOccurTime = cell2mat(countOccurTime);
transMatsTimeFinal1 = cell(1,maxDuration+1);                                   %   A transition matrix for each instant of time is initialized
uniqueFirtZones = unique(list(:,1));
for j = 1:maxDuration+1
    transMatsTimeFinal1{1,j} = zeros(sizeDictionary,sizeDictionary);
    for i = 1:size(uniqueFirtZones,1)
        currIniState = uniqueFirtZones(i,1);
        idTrans = find(list(:,1) == currIniState);
        freqEvents = countOccurTime(idTrans,j);
        for ij = 1:size(idTrans,1)
            if list(idTrans(ij),1) ~= list(idTrans(ij),2)
                transMatsTimeFinal1{1,j}(list(idTrans(ij),1),...
                    list(idTrans(ij),2)) = freqEvents(ij,1);
            else
                if j < maxDuration+1
                    freqModif = countOccurTime(idTrans(ij),j+1);
                    transMatsTimeFinal1{1,j}(list(idTrans(ij),1),...
                        list(idTrans(ij),2)) = freqModif;
                else
                    transMatsTimeFinal1{1,j}(list(idTrans(ij),1),...
                        list(idTrans(ij),2)) = freqEvents(ij,1);
                end
            end
        end
    end
    den = sum(transMatsTimeFinal1{1,j},2);
    den = den + (den==0);
    transMatsTimeFinal1{1,j} = transMatsTimeFinal1{1,j}./repmat(den,1,sizeDictionary);
end