function policy = updatePolicy(MU,explore)

%Choose a random action
policy = MU.A(randi(length(MU.A)));

%With probability explore choose a random policy of length policy
if rand(1) > explore
    for ipolicy = 1:length(policy)
        policy(ipolicy) = MU.A(randi(length(MU.A)));
    end
end