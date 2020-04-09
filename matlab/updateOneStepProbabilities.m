function [MU] = updateOneStepProbabilities(MU)

for m = 1:length(MU.M)
    for a = 1:length(MU.A)
        for m_prime = 1:length(MU.M)
            for a_prime = 1:length(MU.A)
                total = 0;
                for m_prime_prime = 1:length(MU.M)
                    total = total + MU.OneTCounts(m,a,m_prime,a_prime,m_prime_prime);
                end
                for m_prime_prime = 1:length(MU.M)
                    MU.OneT(m,a,m_prime,a_prime,m_prime_prime) = MU.OneTCounts(m,a,m_prime,a_prime,m_prime_prime)/total;
                end
            end
        end
    end
end
            
    