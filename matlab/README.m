%runPSBL.m is the script to setup the environment and run PSBL for sPOMDPs.
%The code was written as directly as possible from the psuedocode presented
%in section 6.3.4 of Dr. Thomas Joseph Collins' Dissertation at the
%University of Southern California in the Computer Science Dept

% 4/9/2020 - STH: Initial repo addition code does not run properly, it has
% errors and the trySplit algorithm has not been coded up.
%Also: important note I took a shortcut by "encoding" observations as
%numbers multiples of 10 and actions multiples of 100 which helps with
%indexing instead of having to search for a string. It is hacky and limits
%the code to a maximum of 9 observations and 9 actions.