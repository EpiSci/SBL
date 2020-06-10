import numpy as np
import copy


# Helper class to be used with the trie in the model
class TrieNode():
    def __init__(self, value=None, lfs=[]):
        self.leaves = lfs
        self.val = value

# Given the head of a trie, find the largest consistent sequence that match one or more of the paths along the trie (See Figure 6.9)
def largestConsistentSequence(head, sequence):
    if not not sequence:
        for leaf in head.leaves:
            if leaf.val == sequence[0]:
                list =  largestConsistentSequence(leaf,sequence[1:])
                list.insert(0,leaf.val)
                return list

    # If it gets to this point, either head has no leaves or the sequence doesn't match
    # Either way, return an empty list
    return []

def insertSequence(head,sequence):
    if not not sequence:
        for leaf in head.leaves:
            if leaf.val == sequence[0]:
                insertSequence(leaf,sequence[1:])
                return None
        # If it gets here, then need to insert a new node
        newNode = TrieNode(sequence[0])
        head.leaves.append(newNode)
        insertSequence(newNode,sequence[1:])
    return None
        

class CollinsModel():
    # Algorithm 11: Initialize sPOMDP model
    def __init__(self, environment, firstObservation, minimumGain):
        self.env = environment
        # Initialize the trie
        leaves = []
        for o in self.env.O_S:
            leaves.append(TrieNode(o))
        self.trieHead = TrieNode(None,leaves)
        
        #The instance variable self.env has a current model associated with it. Thus lines 5 through 14 are unnecessary (lines 12 and 13 will be addressed below).
        #Note: lines 12 and 13 set the belief state to be 1 at the current observation
        self.beliefState = np.zeros([1,len(self.env.O_S)])
        self.beliefState[self.env.O_S.index(firstObservation)] = 1
        
        # Note: self.TCounts is of shape (a,m,m') and not (m,a,m') for efficiency
        self.TCounts = np.ones((len(self.env.A_S),len(self.env.SDE_Set),len(self.env.SDE_Set)))
        #Note: using a (a,a',m,m',m'') matrix instead of a counter list for efficiency
        self.OneTCounts = np.ones((len(self.env.A_S),len(self.env.A_S),len(self.env.SDE_Set),len(self.env.SDE_Set),len(self.env.SDE_Set)))
        #Note: not using M.T, M.OneT, and Algorithm 12 as these can be determined more efficiently by using dirichlet distributions
        self.actionHistory = []
        self.observationHistory = []
        self.observationHistory.append(firstObservation)
        self.beliefHistory = []
        self.beliefHistory.append(copy.deepcopy(self.beliefState))
        self.minGain = minimumGain

    # Reinitialize a model (after the new SDEs have been inserted)
    def reinitializeModel(self):
        self.TCounts = np.ones((len(self.env.A_S),len(self.env.SDE_Set),len(self.env.SDE_Set)))
        self.OneTCounts = np.ones((len(self.env.A_S),len(self.env.A_S),len(self.env.SDE_Set),len(self.env.SDE_Set),len(self.env.SDE_Set)))
        
        self.beliefState = np.zeros([1,len(self.env.O_S)])
        sdeFirstObservations = [sde[0] for sde in self.env.SDE_Set]
        self.beliefState[sdeFirstObservations.index(self.environment.get_observation())] = 1
        self.beliefState = self.beliefState / np.sum(beliefState)
        self.beliefHistory = []
        self.beliefHistory.append(copy.deepcopy(self.beliefState))
        self.actionHistory = []
        self.observationHistory = []

        

# Algorithm 10: PSBL Learning of SPOMDP Models
def psblLearning(env, numActions, explore, patience,minGain):
    prevOb = env.reset()
    model = CollinsModel(env,prevOb,minGain)
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
        #Algorithm 18
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
        updateTransitionFunctionPosteriors(model, a, nextOb)
        # Algorithm 17
        updateOneStepFunctionPosteriors(model, history)
        model.actionHistory.pop(0)
        model.observationHistory.pop(0)

    # Algorithm 14
    model.beliefState = updateBeliefState(model, model.beliefState, a, nextOb)
    model.beliefHistory.append(copy.deepcopy(model.beliefState))
    if len(model.beliefHistory) > len(model.actionHistory):
        model.beliefHistory.pop(0)

# Algorithm 14: sPOMDP Belief Update
def updateBeliefState(model, b, a, o):
    a_index = model.env.A_S.index(a)
    joint = np.zeros([len(b),len(b)])
    for m in range(len(b)):
        for m_prime in range(len(b)):
            joint[m][m_prime] = (dirichlet.mean(model.TCounts[a_index, m, :])[m_prime])*b[m]#model.T[m][a_index][m_prime]*b[m]

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

#Algorithm 16: Transition Function Posteriors Update
def updateTransitionFunctionPosteriors(model, a, o):
    a_index = model.env.A_S.index(a)
    counts = np.zeros([len(model.beliefState),len(model.beliefState)])
    totalCounts = 0
    for (m_idx, m) in enumerate(model.env.SDE_Set):
        for (mp_idx, m_prime) in enumerate(model.env.SDE_Set):
            multFactor = int(m_prime[0] == o)
            counts[m_idx][mp_idx] = multFactor * (dirichlet.mean(model.TCounts[a_index, m_idx, :])[mp_idx]) * beliefHistory[0][m_idx]
            totalCounts = totalCounts + counts[m_idx][mp_idx]

    for m_idx in range(model.beliefStates):
        for mp_idx in range(model.beliefStates):
            counts[m_idx][mp_idx] = counts[m_idx][mp_idx] / totalCounts
            model.TCounts[a_index][m_idx][mp_idx] = model.TCounts[a_index][m_idx][mp_idx] + counts[m_idx][mp_idx]

    #Note: Not necessary to do updateTransitionProbabilities (Algorithm 12) since this is handled by the dirichlet distributions

# Algorithm 17: One Step Transition Function Posteriors Update
def updateOneStepFunctionPosteriors(model, history):
    o = history[0]
    a = history[1]
    o_prime = history[2]
    a_prime = history[3]
    a_index = model.env.A_S.index(a)
    ap_index = model.env.A_S.index(a_prime)
    counts = np.zeros([len(model.beliefState),len(model.beliefState),len(model.beliefState)])
    totalCounts = 0
    for (m_idx,m) in enumerate(model.env.SDE_Set):
        for (mp_idx,mp) in enumerate(model.env.SDE_Set):
            for (mdp_idx,mdp) in enumerate(model.env.SDE_Set):
                multFactor1 = int(mp[0] == o)
                multFactor2 = int(mdp[0] == o_prime)
                counts[m_idx][mp_idx][mdp_idx] = multFactor1 * multFactor2 * (dirichlet.mean(model.TCounts[ap_index, mp_idx, :])[mdp_idx]) * (dirichlet.mean(model.TCounts[a_index, m_idx, :])[mp_idx]) * model.beliefHistory[0][m_idx]
                totalCounts = totalCounts + counts[m_idx][mp_idx][mdp_idx]

    #Note: Not necessary to do updateOneStepProbabilities (analagous to Algorithm 12) since this is handled by the dirichlet distributions


# Algorithm 18: sPOMDP Model State Splitting
def trySplit(model):
    G_ma = computeGains(model)
    G = []
    # Generate the list G that is used to order the model splitting
    mTrajLengths = [len(sde) for sde in model.env.SDE_Set]
    sortedIndexes = sorted(range(len(mTrajLengths),key= mTrajLengths.__getitem__))
    for m in sortedIndexes:
        for a in model.env.A_S:
            # Note: These are only sorted according to the length of m.trajectory, as described in the algorithm (i.e. the sorting with respect to the action a is arbitrary)
            G.append(((model.env.SDE_Set[m],model.env.A_S[a]),G_ma[a][m]))

    for gs in G:
        state = gs[0][0]
        action = gs[0][1]
        gainValue = gs[1]
        if gainValue > model.minGain:
            #Set m1 and m2 to be the two most likely states that are transitioned into from state m taking action a
            m_index = model.env.SDE_Set.index(state)
            a_index = model.env.A_S.index(action)
            transitionSetProbs = dirichlet.mean(model.TCounts[a_index, m_idx, :])
            orderedVals = copy.deepcopy(transitionSetProbs)
            orderedVals.sort()
            prob1 = orderedVals[-1] #largest probability
            prob2 = orderedvals[-2] #second largest probability
            sde1_idx = np.where(transitionSetProbs == prob1)[0][0]
            sde2_idx = np.where(transitionSetProbs == prob2)[0][0]
            m1 = env.get_SDE()[sde1_idx]
            m2 = env.get_SDE()[sde2_idx]

            newOutcome1 = copy.deepcopy(m1)
            newOutcome1.insert(0,action)
            newOutcome1.insert(0,state[0])

            newOutcome2 = copy.deepcopy(m2)
            newOutcome2.insert(0,action)
            newOutcome2.insert(0,state[0])

            outcomesToAdd = []
            if newOutcome1 not in model.env.SDE_Set:
                outcomesToAdd.append(newOutcome1)
                insertSequence(model.trieHead,newOutcome1)

            if newOutcome2 not in model.env.SDE_Set:
                outcomesToAdd.append(newOutcome2)
                insertSequence(model.trieHead,newOutcome2)

            #Note: Not updating model.MaxOutcomeLength as this is generated dynamically when needed in Algorithm 13
            if len(outcomesToAdd) > 1:
                # Note: The modelState class is not used in this implementation so making a new modelState instance is not necessary.

                model.env.SDE_Set.add(newOutcome1)
                model.env.SDE_Set.add(newOutcome2)
                model.reinitializeModel()
                return True
            
            elif len(outcomesToAdd) == 1:
                model.env.SDE_Set.add(outcomesToAdd[0])
                model.reinitializeModel()
                return True
            
    return False

# Compute gains according to equation 6.10 (Helper Function for Algorithm 18)
def computeGains(model):
    G = np.zeros([len(model.env.A_S),len(model.env.SDE_Set)])

    for mp in range(len(model.env.SDE_Set)):
        for ap in range(len(model.env.SDE_Set)):
            sum = 0
            for m in range(len(model.env.SDE_Set)):
                for a in range(len(model.env.SDE_Set)):
                    w_ma = (dirichlet.mean(model.TCounts[a, m, :])[mp])
                    sum = sum + (w_ma * entropy((dirichlet.mean(model.OneTCounts[a, ap, m, mp, :])), base=len(model.env.SDE_Set)))
            G[ap][mp] = entropy((dirichlet.mean(model.TCounts[ap, mp, :])), base=len(model.env.SDE_Set)) - sum
    return G
