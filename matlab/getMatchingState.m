function [state] = getMatchingState(stringsToState,match)

for istate = 1:length(stringsToState)
    if strcmp(stringToState{istate}.strVal,match)
        state = stringToState{istate}.state;
    end
end