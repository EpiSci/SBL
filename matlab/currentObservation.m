function [o] = currentObservation(E)
    o = E.s{E.c_state}.o;
end

