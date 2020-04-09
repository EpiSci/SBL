function [MU] = updateOneStepFunctionPosteriors(history,MU) %algorithm 17
o = history(1);
a = history(2)/100;
o_prime = history(3);
a_prime = history(4)/100;

counts = zeros(length(MU.M),length(MU.M),length(MU.M));

totalCounts = 0;


for m = 1:length(MU.M)
    for m_prime = 1:length(MU.M)
        for m_prime_prime = 1:length(MU.M)
            firstObservation = MU.M{m_prime}.trajectory;
            multFactor1 = double(firstObservation == o);
            firstObservation = MU.M{m_prime_prime}.trajectory;
            multFactor2 = double(firstObservation == o_prime);
            counts(m,m_prime,m_prime_prime) = multFactor2 * multFactor1*MU.T(m_prime,a_prime,m_prime_prime)*MU.T(m,a,m_prime)*MU.beliefHistory{1}(m);
            totalCounts = totalCounts + counts(m,m_prime,m_prime_prime);
        end
    end
end

for m = 1:length(MU.M)
    for m_prime = 1:length(MU.M)
        for m_prime_prime = 1:length(MU.M)
            counts(m,m_prime,m_prime_prime) = counts(m,m_prime,m_prime_prime)/ totalCounts;
            MU.OneTCounts(m,a,m_prime,a_prime,m_prime_prime) = MU.OneTCounts(m,a,m_prime,a_prime,m_prime_prime) + counts(m,m_prime,m_prime_prime);
        end
    end
end

MU = updateOneStepProbabilities(MU);
