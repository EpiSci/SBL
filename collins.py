import numpy as np
import copy

class CollinsModel():
    #Algorithm 11: Initialize sPOMDP model
    def __init__(self, environment, firstObservation):
        self.env = environment
        #Note: not using a trie; instead the model's SDEs can be used
        #The instance variable self.env has a current model associated with it. Thus lines 5 through 14 are unnecessary (lines 12 and 13 will be addressed below).
        #Note: lines 12 and 13 set the belief state to be 1 at the current observation
        self.beliefState = np.zeros([1,len(env.O_S)])
        self.beliefState[env.O_S.index(firstObservation)] = 1
        
        #Note: self.TCounts is of shape (a,m,m') and not (m,a,m') for efficiency
        self.TCounts = np.ones((len(env.A_S),len(SDE_List),len(SDE_List)))
        #Note: using a (a,a',m,m',m'') matrix instead of a counter list for efficiency
        self.OneTCounts = np.ones((len(env.A_S),len(env.A_S),len(SDE_List),len(SDE_List),len(SDE_List)))
        #Note: not using M.T, M.OneT, and Algorithm 12 as these can be determined more efficiently by using dirichlet distributions
        self.actionHistory = []
        self.observationHistory = []
        self.observationHistory.append(firstObservation)
        self.beliefHistory = []
        self.beliefHistory.append(copy.deepcopy(self.beliefState))
        
        

# Algorithm 10: PSBL Learning of SPOMDP Models
def psblLearning(env, numActions, explore, patience):
    prevOb = env.reset()
    model = CollinsModel(env,prevOb)
    minSurpriseModel = None
    minSurprise = float("inf")
    splitsSinceMin = 0
    policy = []
    foundSplit = True
    while foundSplit:
        for i in range(numActions):
            if not policy:
                # Add actions of an SDE to the policy or random actions
                policy = updatePolicy(env, explore)
            action = policy.pop()
            nextOb = env.step(action)
            # Algorithm 13:
            updateModelParameters(env, action, prevOb, nextOb)
            prevOb = nextOb

        newSurprise = computeSurprise(env)
        if newSuprise < minSurprise:
            minSurprise = newSurprise
            minSurpriseModel = copy.deepcopy(env)
            splitsSinceMin = 0
        else:
            splitsSinceMin = 1
        if splitsSinceMin > patience:
            break
        foundSplit = trySplit(env)
    return minSurpriseModel


