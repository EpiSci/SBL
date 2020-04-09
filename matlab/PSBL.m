function [MU] = PSBL(numActions,explore,A,O,patience,E) %Algorithm 10
MU = initializeModel(A,O,E); % Algorhtim 11
minSurpriseModel = [];
minSurprise = inf;
splitsSinceMin = 0;
foundSplit = 1;

policy = [];

while foundSplit
    for iA = 1:numActions
        if isempty(policy)
            policy = updatePolicy(MU,explore);
        end
        
        action = policy(1);
        if length(policy) > 1
            policy = policy(2:end);
        else
            policy = [];
        end
        
        prevOb = currentObservation(E);
        [E,nextOb] = takeAction(E,action);
        MU = updateModelParameters(MU, action, prevOb, nextOb); %Algorithm 13
    end
    newSurprise = computeSurprise(MU);
    
    if newSurprise < minSurprise
        minSurprise = newSurprise;
        minSurpriseModel = MU;
        splitsSinceMin = 0;
    else
        splitsSinceMin = splitsSinceMin + 1;
    end
    
    if splitsSinceMin > patience
        break
    end
    
    foundSplit = trySplit(MU);   
end

