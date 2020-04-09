function [MU] = updateBeliefState(MU, action, nextOb)
b = MU.beliefState;
a = action/100;

joint = zeros(length(MU.M),length(MU.M));

for m = 1:length(MU.M)
    for m_prime = 1:length(MU.M)
        joint(m,m_prime) = MU.T(m, a, m_prime)*b(m);
    end
end

b_prime = zeros(1,length(MU.M));

for m = 1:length(MU.M)
    for m_prime = 1:length(MU.M)
        b_prime(m_prime) = b_prime(m_prime) + joint(m,m_prime);
    end
end

for m = 1:length(MU.M)
    firstObservation = MU.M{m}.trajectory;
    multFactor = double(firstObservation == nextOb);
    b_prime(m) = b_prime(m)*multFactor;
end

total = 0;

for m = 1:length(MU.M)
    total = total + b_prime(m);
end


for m = 1:length(MU.M)
    b_prime(m) = b_prime(m)/total;
end

MU.beliefState = b_prime;
end