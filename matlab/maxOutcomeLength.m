function maxValue = maxOutcomeLength(MU)

maxValue = -inf;
for m = 1:length(MU.M)
    if length(MU.M{1}.trajectory) > maxValue
        maxValue = length(MU.M{1}.trajectory);
    end
end

