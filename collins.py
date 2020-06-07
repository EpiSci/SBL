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
            updateModelParameters(model, action, prevOb, nextOb)
            prevOb = nextOb

        newSurprise = computeSurprise(env)
        if newSuprise < minSurprise:
            minSurprise = newSurprise
            minSurpriseModel = copy.deepcopy(model)
            splitsSinceMin = 0
        else:
            splitsSinceMin = 1
        if splitsSinceMin > patience:
            break
        foundSplit = trySplit(model)
    return minSurpriseModel


#Algorithm 13: Update sPOMDP Model Parameters
def updateModelParameters(model, action, prevOb, nextOb):
    model.actionHistory.append(a)
    model.observationHistory.append(nextOb)
    history = [val for pair in zip(model.actionHistory,model.observationHistory) for val in pair]
    #Note: the previous line will only work for lists of the same length. Since the observation history has one more element, we need to append the nextOb to the end of the history
    history.append(nextOb)
    maxOutcomeLength = max([len(sde) for sde in model.env.SDE_Set])
    if len(history) > maxOutcomeLength + 6:
        # Algorithm 15
        model.beliefHistory = smoothBeliefHistory(history, model.beliefHistory)
        # Algorithm 16
        updateTransitionFunctionPosteriors(a, nextOb, model.beliefHistory)
        # Algorithm 17
        updateOneStepFunctionPosteriors(history, model.beliefHistory)
        model.actionHistory.pop(0)
        model.observationHistory.pop(0)

    # Algorithm 14
    model.beliefState = updateBeliefState(model.beliefState, a, nextOb)
    model.beliefHistory.append(copy.deepcopy(model.beliefState))
    if len(model.beliefHistory) > len(model.actionHistory):
        model.beliefHistory.pop(0)
