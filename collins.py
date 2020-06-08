import numpy as np
import copy

#Helper class to be used with the trie in the model
class TrieNode():
    def __init__(self,value=None,lfs=[]):
        self.leaves = lfs
        self.val = value

#Given the head of a trie, find the largest consistent sequence that match one or more of the paths along the trie (See Figure 6.9)
def largestConsistentSequence(head,sequence):
    if not not sequence:
        for leaf in head.leaves:
            if leaf.val == sequence[0]:
                list =  largestConsistentSequence(leaf,sequence[1:])
                list.insert(0,leaf.val)
                return list

    #If it gets to this point, either head has no leaves or the sequence doesn't match
    #Either way, return an empty list
    return []
        

class CollinsModel():
    #Algorithm 11: Initialize sPOMDP model
    def __init__(self, environment, firstObservation):
        self.env = environment
        #Initialize the trie
        leaves = []
        for o in self.env.O_S:
            leaves.append(TrieNode(o))
        self.trieHead = TrieNode(None,leaves)
        
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
        model.beliefHistory = smoothBeliefHistory(model,history, model.beliefHistory)
        # Algorithm 16
        updateTransitionFunctionPosteriors(a, nextOb, model.beliefHistory)
        # Algorithm 17
        updateOneStepFunctionPosteriors(history, model.beliefHistory)
        model.actionHistory.pop(0)
        model.observationHistory.pop(0)

    # Algorithm 14
    model.beliefState = updateBeliefState(model, model.beliefState, a, nextOb)
    model.beliefHistory.append(copy.deepcopy(model.beliefState))
    if len(model.beliefHistory) > len(model.actionHistory):
        model.beliefHistory.pop(0)

# Algorithm 14: sPOMDP Belief Update
def updateBeliefState(model, b, a, o):
    a_index = index(model.env.A_S)
    joint = np.zeros([len(b),len(b)])
    for m in range(len(b)):
        for m_prime in range(len(b)):
            joint[m][m_prime] = model.T[m][a_index][m_prime]*b[m] #This line will throw an error. We need to use the Dirichlet distributions since we don't formally have a T matrix

    b_prime = [0 for val in range(m)]
    for m_prime in range(len(b)):
        for m in range(len(b)):
            b_prime[m_prime] = b_prime[m_prime] + joint[m][m_prime]

    for (m_idx, m) in enumerate(model.env.SDE_List):
        multFactor = int(m[0] == o)
        b_prime[m_idx] = b_prime[m_idx]*multFactor

    total = 0
    for m in range(len(b)):
        total = total + b_prime[m]
    for m in rnage(len(b)):
        b_prime[m] = b_prime[m] / total
    return b_prime


            
#Algorithm 15: Smooth Belief History
def smoothBeliefHistory(model, history, beliefHistory):
    for i in range(3):
        savedBeliefs = copy.deepcopy(beliefHistory[i])
        largestMatching = largestConsistentSequence(model.trieHead,history[2*i:])
        matching = [sde for sde in model.env.SDE_Set if sde[0:len(largestMatching)] == largestMatching] #Only include those SDEs that contain thte largestMatching sequence at their beginning
        beliefHistory[i] = [0 for val in range(len(beliefHistory[i]))]
        for match in matching:
            matchingState = model.env.SDE_Set.index(match)
            beliefHistory[i][matchingState] = savedBeliefs[matchingState]

        total = 0
        for m in len(model.env.SDE_Set):
            total = total + beliefHistory[i][m]
        for m in len(model.env.SDE_Set):
            beliefHistory[i][m] = beliefHistory[i][m] / total

    return beliefHistory
            
        


# print("hi")
# n8 = TrieNode("diamond",[])
# n7 = TrieNode("square",[])
# n6 = TrieNode("diamond",[])
# n5 = TrieNode("x",[n7,n8])
# n4 = TrieNode("y",[n6])
# n3 = TrieNode("diamond",[n5])
# n2 = TrieNode("square",[n4])
# n1 = TrieNode(None, [n2,n3])
# print(largestConsistentSequence(n1,["square","y","diamond"]))
# print(largestConsistentSequence(n1,["chartrues"]))
