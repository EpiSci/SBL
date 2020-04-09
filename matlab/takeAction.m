function [E, nextObs] = takeAction(E,a)
    action = a/100; % actions are "encoded" by * 100
    nextObs = E.s{E.c_state}.a{action}.o;
    E.c_state = E.s{E.c_state}.a{action}.s;
end