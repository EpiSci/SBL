function [POMDP] = initalizeState(POMDP)
    POMDP.c_state = randi(length(POMDP.s));
end

