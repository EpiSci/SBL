%Environment and run script
clear all;
close all;
POMDP.s{1}.o = 10;
POMDP.s{1}.a{1}.s = 2;
POMDP.s{1}.a{1}.o = 10;
POMDP.s{1}.a{1}.v = 0.99;
POMDP.s{1}.a{2}.s = 3;
POMDP.s{1}.a{2}.o = 20;
POMDP.s{1}.a{2}.v = 0.99;

POMDP.s{2}.o = 10;
POMDP.s{2}.a{1}.s = 2;
POMDP.s{2}.a{1}.o = 10;
POMDP.s{2}.a{1}.v = 0.99;
POMDP.s{2}.a{2}.s = 4;
POMDP.s{2}.a{2}.o = 20;
POMDP.s{2}.a{2}.v = 0.99;

POMDP.s{3}.o = 20;
POMDP.s{3}.a{1}.s = 1;
POMDP.s{3}.a{1}.o = 10;
POMDP.s{3}.a{1}.v = 0.99;
POMDP.s{3}.a{2}.s = 1;
POMDP.s{3}.a{2}.o = 10;
POMDP.s{3}.a{2}.v = 0.99;

POMDP.s{4}.o = 20;
POMDP.s{4}.a{1}.s = 3;
POMDP.s{4}.a{1}.o = 20;
POMDP.s{4}.a{1}.v = 0.99;
POMDP.s{4}.a{2}.s = 2;
POMDP.s{4}.a{2}.o = 10;
POMDP.s{4}.a{2}.v = 0.99;

POMDP = initalizeState(POMDP);

O = [10 20]; % square = 10; diamond = 20
A = [100 200]; % x = 100; y = 200
numActions = 1000;
explore = 0.5;
patience = 50000;
[MU] = PSBL(numActions,explore,A,O,patience,POMDP);