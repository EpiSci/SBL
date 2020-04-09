function [MU] = updateModelParameters(MU,a, prevOb, nextOb) %Algorithm 13

MU.actionHistory(end+1) = a;
MU.observationHistory(end+1) = nextOb;
history = zeros(1,length(MU.actionHistory)+length(MU.observationHistory));
history(1:2:end) = MU.observationHistory; % had to reverse the order of the obs and action since history always starts with obs in initialization
history(2:2:end) = MU.actionHistory;

if length(history) >= maxOutcomeLength(MU) + 6
    MU = smoothBeliefHistory(history, MU); %Algorithm 15
    MU = updateTransitionFunctionPosteriors(a, nextOb, MU); %Algorithm 16
    MU = updateOneStepFunctionPosteriors(history,MU); %Algorithm 17
    MU.actionHistory = MU.actionHistory(2:end); %popleft
    MU.observationHistory= MU.observationHistory(2:end);
end

%Algorithm 14
MU = updateBeliefState(MU, a, nextOb);
MU.beliefHistory{end+1} = MU.beliefState;
if size(MU.beliefHistory) > size(MU.actionHistory)
    MU.beliefHistory = MU.beliefHistory{2:end}; %popleft
end