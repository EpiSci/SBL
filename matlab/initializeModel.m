function [MU] = initializeModel(A,O,E)
MU.sPOMDPModel = [];
MU.outcomeTrie = [];
MU.beliefState = zeros(1,length(O));
MU.A = A;

stateCount = 1;
for o = O
    trajectory = o;
    newState.ids = stateCount;
    newState.trajectory = trajectory;
    MU.M{stateCount} = newState;
    MU.outcomeTrie{stateCount} = trajectory;
    MU.idsToStates{stateCount} = newState;
    MU.stringsToState{stateCount}.strVal = num2str(trajectory);
    MU.stringsToState{stateCount}.state = newState;
    if o == currentObservation(E)
        MU.beliefState(stateCount) = 1;
    end
    stateCount = stateCount + 1;
end
 
lenA = length(A);
lenM = length(MU.M);

 MU.TCounts = ones(lenM,lenA,lenM);
 MU.T = zeros(lenM,lenA,lenM);
 MU.OneTCounts = ones(lenM,lenA,lenM,lenA,lenM);
 MU.OneT = zeros(lenM,lenA,lenM,lenA,lenM);
 
MU = updateTransitionProbabilities(MU);
MU = updateOneStepProbabilities(MU);
MU.actionHistory = [];
MU.observationHistory = [currentObservation(E)];
MU.beliefHistory{1} = MU.beliefState;
 
 
 

