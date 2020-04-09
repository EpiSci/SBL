function [matching] = consistentSequences(outcomeTrie,history)

matching = {};

for i = 1:length(outcomeTrie)
    cTrie = outcomeTrie{i};
    for j = 1:(length(history)-length(cTrie))
        if sum(history(j:(j+length(cTrie))) - cTrie) == 0
            matching{end+1} = cTrie;
        end
    end
end
        
