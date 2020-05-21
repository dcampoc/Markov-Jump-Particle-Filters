function distances = distance8(currState,averageMat)
currS1 = currState(1,1); currS2 = currState(1,2);
currS3 = currState(1,3); currS4 = currState(1,4);
currS5 = currState(1,5); currS6 = currState(1,6);
currS7 = currState(1,7); currS8 = currState(1,8);
protM1 = averageMat(:,1); protM2 = averageMat(:,2);
protM3 = averageMat(:,3); protM4 = averageMat(:,4);
protM5 = averageMat(:,5); protM6 = averageMat(:,6);
protM7 = averageMat(:,7); protM8 = averageMat(:,8);

dist1 = abs(protM1 - currS1);
dist2 = abs(protM2 - currS2);
dist3 = abs(protM3 - currS3);
dist4 = abs(protM4 - currS4);
dist5 = abs(protM5 - currS5);
dist6 = abs(protM6 - currS6);
dist7 = abs(protM7 - currS7);
dist8 = abs(protM8 - currS8);
distances = [dist1 dist2 dist3 dist4 dist5 dist6 dist7 dist8];

end