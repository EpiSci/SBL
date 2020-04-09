function [MU] = smoothBeliefHistory(history, MU) %Algorithm 15

for ii = 1:3
    savedBeliefs = MU.beliefHistory{ii};
    matching = consistentSequences(MU.outcomeTrie,history((2*(ii-1)+1):end));
    beliefHistory{ii} = zeros(1,length(MU.M));
    
    for imatch = 1:length(matching)
        match = num2str(matching{imatch});
        matchingState = getMatchingState(MU.stringsToStates,match);
        beliefHistory{ii}(matchingState.ids) = savedBeliefs(matchingState.ids);
    end
    
    total = 0;
    for m = 1:length(MU.M)
        total = total+ beliefHistory{ii}(m);
    end
    
    for m = 1:length(MU.M)
        beliefHistory{ii}(m) = beliefHistory{ii}(m)/total;
    end
end
MU.beliefHistory = beliefHistory;
    
    

