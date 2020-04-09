function [MU] = updateTransitionFunctionPosteriors(action, nextOb, MU) %algorithm 16
counts = zeros(length(MU.M),length(MU.M));
a = action/100;

totalCounts = 0;
beliefHistory = MU.beliefHistory;

for m = 1:length(MU.M)
    for m_prime = 1:length(MU.M)
        firstObservation = MU.M{m_prime}.trajectory;
        multFactor = double(firstObservation == nextOb);
        counts(m,m_prime) = multFactor*MU.T(m,a,m_prime)*beliefHistory{2}(m);
        totalCounts = totalCounts +  counts(m,m_prime);
    end
end

for m = 1:length(MU.M)
    for m_prime = 1:length(MU.M)
        counts(m,m_prime) = counts(m,m_prime)/totalCounts;
        MU.TCounts(m,a,m_prime) = MU.TCounts(m,a,m_prime) + counts(m,m_prime);
    end
end

MU = updateTransitionProbabilities(MU);


