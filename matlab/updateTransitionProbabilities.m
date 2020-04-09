function [MU] = updateTransitionProbabilities(MU)

for m = 1:length(MU.M)
    for a = 1:length(MU.A)
        total = 0;
        for m_prime = 1:length(MU.M)
            total = total + MU.TCounts(m,a,m_prime);
        end
        for m_prime = 1:length(MU.M)
            MU.T(m,a,m_prime) = MU.TCounts(m,a,m_prime)/total;
        end
    end
end
            
    