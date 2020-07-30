import numpy as np
import copy
from scipy.stats import dirichlet, entropy
import random
import csv
import git
import pomdp
import test
import networkx as nx
from anytree import Node, LevelOrderGroupIter
from anytree.exporter import UniqueDotExporter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from subprocess import check_call


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
        newNode = TrieNode(sequence[0],[])
        head.leaves.append(newNode)
        insertSequence(newNode,sequence[1:])
        return None
    return None

def printTrie(head):
    print("Head node is " + str(head.val))
    for (i,leaf) in enumerate(head.leaves):
        print("Child " + str(i) + " of head " + str(head.val) + " has value " + str(leaf.val))
        printTrie(leaf)
        

class CollinsModel():
    # Algorithm 11: Initialize sPOMDP model
    def __init__(self, environment, firstObservation, minimumGain):
        self.env = environment
        # Initialize the trie
        self.trieHead = TrieNode(None,[])
        for sde in self.env.SDE_Set:
            insertSequence(self.trieHead,sde)

        
        #The instance variable self.env has a current model associated with it. Thus lines 5 through 14 are unecessary (lines 12 and 13 will be addressed below).
        #Note: lines 12 and 13 set the belief state to be 1 at the current observation
        sdeFirstObservations = [sde[0] for sde in self.env.SDE_Set]
        self.beliefState = [1 if val == firstObservation else 0 for val in sdeFirstObservations]
        self.beliefState = self.beliefState / np.sum(self.beliefState)
        
        # Note: self.TCounts is of shape (a,m,m') and not (m,a,m') for consistency
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

        self.endEarly = False


    # Reinitialize a model (after the new SDEs have been inserted)
    def reinitializeModel(self):
        self.TCounts = np.ones((len(self.env.A_S),len(self.env.SDE_Set),len(self.env.SDE_Set)))
        self.OneTCounts = np.ones((len(self.env.A_S),len(self.env.A_S),len(self.env.SDE_Set),len(self.env.SDE_Set),len(self.env.SDE_Set)))
        self.beliefState = np.zeros([1,len(self.env.O_S)])
        sdeFirstObservations = [sde[0] for sde in self.env.SDE_Set]
        self.beliefState = [1 if val == self.observationHistory[-1] else 0 for val in sdeFirstObservations]
        self.beliefState = self.beliefState / np.sum(self.beliefState)
        self.beliefHistory = []
        self.beliefHistory.append(copy.deepcopy(self.beliefState))
        self.actionHistory = []
        prevOb = self.observationHistory[-1]
        self.observationHistory = [prevOb]

        

# Algorithm 10: PSBL Learning of SPOMDP Models
def psblLearning(env, numActions, explore, patience,minGain, insertRandActions, writeToFile, filename,useBudd, revisedSplitting, haveControl, confidence_factor, localization_threshold):

    prevOb = env.reset()
    model = CollinsModel(env,prevOb,minGain)
    minSurpriseModel = None
    minSurprise = float("inf")
    splitsSinceMin = 0
    policy = []
    foundSplit = True
    modelNum = 0

    if writeToFile:
        c = csv.writer(open(filename, "w", newline=''))
        c.writerow([x for x in range(30)])
        
        # Write git repo sha
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        c.writerow(["github Code Version (SHA):", sha])
        
        # Write the training parameters into the .csv file
        parameterNames = ["Environment Observations","Environment Actions","alpha","epsilon", "numActions","explore","patience","gainThresh", "insertRandActions","useBudd", "revisedSplitting", "haveControl", "confidence_factor"]
        parameterVals = [model.env.O_S, model.env.A_S, model.env.Alpha, model.env.Epsilon, numActions, explore, patience, minGain, insertRandActions, useBudd, revisedSplitting, haveControl, confidence_factor]

        c.writerow(parameterNames)
        c.writerow(parameterVals)

    genTraj = False
    useTraj = False
    if genTraj:
        traj = np.zeros([numActions,1],dtype=object)

    minSurpriseModelNum = 0
    
    while foundSplit:
        if useTraj:
            numpyTraj = np.load("Testing Data/traj" + str(modelNum) + ".npy")
            policy = [numpyTraj[i][0] for i in range(numpyTraj.size)]

        performed_experiment = False

        for i in range(numActions):

            if confidence_factor is not None:
                # if we have performed all experiments or there is no place that we can reliably get to do perform an experiment then terminate
                if ((np.min(np.sum(model.TCounts, axis=2)) / len(model.env.SDE_Set)) >= confidence_factor) or (haveControl is True and getPathToExperiment(model, calcTransitionProbabilities(model), np.argmax(model.beliefState), confidence_factor, localization_threshold) is None and np.max(computeGains(model)) > model.minGain):
                    print("Finished early on iteration number " + str(i))
                    if((np.min(np.sum(model.TCounts, axis=2)) / len(model.env.SDE_Set)) >= confidence_factor):
                        print("Performed all necessary experiments")
                    else:
                        print("Couldn't reach all states confidently to perform all of the requested experiments")
                    model.endEarly = False
                    if writeToFile:
                        modelTransitionProbs = calcTransitionProbabilities(model)
                        iterError = pomdp.calculateError(model.env, modelTransitionProbs, 10000, model.TCounts)
                        iterAbsError = pomdp.calculateAbsoluteError(model.env, modelTransitionProbs)
                        c.writerow(["Iteration: ", i])
                        c.writerow(["Error:", iterError])
                        c.writerow(["Absolute Error:", iterAbsError])
                        c.writerow(["Transition Probabilities"])
                        test.writeNumpyMatrixToCSV(c, modelTransitionProbs)
                    break

            if i % 1000 == 0:
                print(i)
            if i % 2500 == 0 or i == numActions - 1:
                if i == 0:
                    c.writerow([])
                    c.writerow(["Model Num " + str(modelNum)])
                    c.writerow(["Model States: "])
                    c.writerow(env.SDE_Set)
                modelTransitionProbs = calcTransitionProbabilities(model)
                iterError = pomdp.calculateError(model.env, modelTransitionProbs, 10000, model.TCounts)
                iterAbsError = pomdp.calculateAbsoluteError(model.env, modelTransitionProbs)
                c.writerow(["Iteration: ", i])
                c.writerow(["Error:", iterError])
                c.writerow(["Absolute Error:", iterAbsError])
                c.writerow(["Transition Probabilities"])
                test.writeNumpyMatrixToCSV(c, modelTransitionProbs)
                c.writerow(["Transition Gammas"])
                test.writeNumpyMatrixToCSV(c, model.TCounts)
                
            if not policy:
                # Add actions of an SDE to the policy or random actions. This will also add a random action between SDEs if insertRandActions is enabled
                (policy, performed_experiment) = updatePolicy(model, explore, prevOb, insertRandActions, haveControl, confidence_factor, performed_experiment, localization_threshold)
                if genTraj:
                    for (actionIdx,action) in enumerate(policy):
                        if actionIdx + i < numActions:
                            traj[i + actionIdx] = action
            action = policy.pop(0)
            nextOb = model.env.step(action)
            # Algorithm 13:
            updateModelParameters(model, action, prevOb, nextOb, useBudd)
            prevOb = nextOb
        if genTraj:
            np.save("Testing Data/traj" + str(modelNum) + ".npy",traj)

        newSurprise = computeSurprise(model)
        print("Transition Probabilities:")
        print(calcTransitionProbabilities(model))
        print("Surprise:")
        print(newSurprise)
        if newSurprise < minSurprise:
            minSurpriseModelNum = modelNum
            minSurprise = newSurprise
            minSurpriseModel = copy.deepcopy(model)
            splitsSinceMin = 0
        else:
            splitsSinceMin = splitsSinceMin + 1
        if splitsSinceMin > patience:
            print("TCounts: ")
            print(model.TCounts)
            print("Stopped model splitting due to a lack of patience.")
            break
        #Algorithm 18
        foundSplit = trySplit(model, revisedSplitting)
        
        modelNum = modelNum + 1
    #make it clear which model was returned
    c.writerow("***********")
    c.writerow(["Model Num " + str(minSurpriseModelNum)])
    c.writerow(["Model States: "])
    c.writerow(minSurpriseModel.env.SDE_Set)
    modelTransitionProbs = calcTransitionProbabilities(minSurpriseModel)
    iterError = pomdp.calculateError(minSurpriseModel.env, modelTransitionProbs, 10000, minSurpriseModel.TCounts)
    iterAbsError = pomdp.calculateAbsoluteError(minSurpriseModel.env, modelTransitionProbs)
    c.writerow(["Iteration: ", i])
    c.writerow(["Error:", iterError])
    c.writerow(["Absolute Error:", iterAbsError])
    c.writerow(["Transition Probabilities"])
    test.writeNumpyMatrixToCSV(c, modelTransitionProbs)
    c.writerow(["Transition Gammas"])
    test.writeNumpyMatrixToCSV(c, minSurpriseModel.TCounts)
    
    return minSurpriseModel

# Helper Function for Algorithm 10
def updatePolicy(model,explore,prevObservation,insertRandActions, haveControl, confidence_factor, performed_experiment, localization_threshold):

    if haveControl is True:
        if getPathToExperiment(model, calcTransitionProbabilities(model), np.argmax(model.beliefState), confidence_factor, localization_threshold) is not None:
            return haveControlPolicy(model, prevObservation, confidence_factor, performed_experiment, localization_threshold)
        else:
            (temp, _) = updatePolicy(model, explore, prevObservation, insertRandActions, False, confidence_factor, performed_experiment, localization_threshold)
            return (temp, performed_experiment)

    random_sample = np.random.random()
    matchingSDEs = model.env.get_SDE(prevObservation)
    randSDE = random.choice(matchingSDEs)
    policy = randSDE[1::2] # Need to pull every other value since both observations and actions are stored in the SDE, but only a policy should be returned
    if insertRandActions:
        policy.append(random.choice(model.env.A_S))
    
    if random_sample > explore and not not policy:
        return (policy, performed_experiment)
    else:
        return (random.choices(model.env.A_S, k=max(1,len(policy))), performed_experiment) # Use max of 1 or the policy length to make sure at least one action is returned

# Helper Function for Algorithm 10
def computeSurprise(model):
    zetas = np.sum(model.TCounts,axis=2) #A AxM matrix
    psi = np.sum(zetas)
    surprise = 0
    for m in range(len(model.env.SDE_Set)):
        for a in range(len(model.env.A_S)):
            surprise = surprise + ((zetas[a][m] / psi)*(entropy((dirichlet.mean(model.TCounts[a, m, :])), base=len(model.env.SDE_Set))))
    return surprise


#Algorithm 13: Update sPOMDP Model Parameters
def updateModelParameters(model, a, prevOb, nextOb, useBudd):
    model.actionHistory.append(a)
    model.observationHistory.append(nextOb)
    history = [val for pair in zip(model.observationHistory,model.actionHistory) for val in pair]
    #Note: the previous line will only work for lists of the same length. Since the observation history has one more element, we need to append the nextOb to the end of the history
    history.append(nextOb)
    maxOutcomeLength = max([len(sde) for sde in model.env.SDE_Set])
    if len(history) >= maxOutcomeLength + 6:
        # Algorithm 15
        model.beliefHistory = smoothBeliefHistory(model,history, model.beliefHistory)
        # Algorithm 16
        # updateTransitionFunctionPosteriors(model, a, nextOb)
        updateTransitionFunctionPosteriors(model, model.actionHistory[0], model.observationHistory[1],useBudd)
        # Algorithm 17
        updateOneStepFunctionPosteriors(model, history, useBudd)
        model.actionHistory.pop(0)
        model.observationHistory.pop(0)

    # Algorithm 14
    model.beliefState = updateBeliefState(model, model.beliefState, a, nextOb)
    model.beliefHistory.append(copy.deepcopy(model.beliefState))
    if len(model.beliefHistory) > len(model.actionHistory) + 1:
        model.beliefHistory.pop(0)

# Algorithm 14: sPOMDP Belief Update
def updateBeliefState(model, b, a, o):
    a_index = model.env.A_S.index(a)
    joint = np.zeros([len(b),len(b)])
    for m in range(len(b)):
        for m_prime in range(len(b)):
            joint[m][m_prime] = (dirichlet.mean(model.TCounts[a_index, m, :])[m_prime])*b[m]#model.T[m][a_index][m_prime]*b[m]

    b_prime = np.zeros(len(b))
    for m_prime in range(len(b)):
        for m in range(len(b)):
            b_prime[m_prime] = b_prime[m_prime] + joint[m][m_prime]

    for (m_idx, m) in enumerate(model.env.SDE_Set):
        multFactor = int(m[0] == o)
        b_prime[m_idx] = b_prime[m_idx]*multFactor

    total = 0
    for m in range(len(b)):
        total = total + b_prime[m]
    for m in range(len(b)):
        b_prime[m] = b_prime[m] / total
    return b_prime


            
#Algorithm 15: Smooth Belief History
def smoothBeliefHistory(model, history, beliefHistory):
    for i in range(3):
        savedBeliefs = copy.deepcopy(beliefHistory[i])
        largestMatching = largestConsistentSequence(model.trieHead,history[2*i:])
        matching = [sde for sde in model.env.SDE_Set if sde[0:len(largestMatching)] == largestMatching] #Only include those SDEs that contain the largestMatching sequence at their beginning
        beliefHistory[i] = np.zeros(len(beliefHistory[i]))
        for match in matching:
            matchingState = model.env.SDE_Set.index(match)
            beliefHistory[i][matchingState] = savedBeliefs[matchingState]

        total = 0
        for m in range(len(model.env.SDE_Set)):
            total = total + beliefHistory[i][m]

        for m in range(len(model.env.SDE_Set)):
            beliefHistory[i][m] = beliefHistory[i][m] / total
            
    return beliefHistory

#Algorithm 16: Transition Function Posteriors Update
def updateTransitionFunctionPosteriors(model, a, o, useBudd):
    a_index = model.env.A_S.index(a)
    counts = np.zeros([len(model.beliefState),len(model.beliefState)])
    totalCounts = 0
    for (m_idx, m) in enumerate(model.env.SDE_Set):
        for (mp_idx, m_prime) in enumerate(model.env.SDE_Set):
            # multFactor = int(m_prime[0] == o)
            multFactor = model.beliefHistory[1][mp_idx] #Note: this is an alternative way of calculating multFactor that is supposed to be better in practice. See Section 6.3.4.3 in Collins' dissertation.
            counts[m_idx][mp_idx] = multFactor * (dirichlet.mean(model.TCounts[a_index, m_idx, :])[mp_idx]) * model.beliefHistory[0][m_idx]
            totalCounts = totalCounts + counts[m_idx][mp_idx]

    if useBudd:
        max_rows = np.argwhere(np.array(model.beliefHistory[0]) == np.amax(np.array(model.beliefHistory[0])))
        if max_rows.size != 1:
            counts[:,:] = 0
        else:
            max_row = max_rows[0]
            counts[np.arange(len(model.env.SDE_Set)) != max_row, :] = 0

        if totalCounts == 0:
            print(counts)
            print(max_row)
            print(model.beliefHistory[0])
            print(model.beliefHistory[1])
            print(counts)
            print(a)
            print(o)
            print(model.env.SDE_Set)
            print(model.TCounts)
            exit()


    for m_idx in range(len(model.beliefState)):
        for mp_idx in range(len(model.beliefState)):
            counts[m_idx][mp_idx] = counts[m_idx][mp_idx] / totalCounts
            model.TCounts[a_index][m_idx][mp_idx] = model.TCounts[a_index][m_idx][mp_idx] + counts[m_idx][mp_idx]

    #Note: Not necessary to do updateTransitionProbabilities (Algorithm 12) since this is handled by the dirichlet distributions

# Algorithm 17: One Step Transition Function Posteriors Update
def updateOneStepFunctionPosteriors(model, history, useBudd):
    o = history[0]
    a = history[1]
    o_prime = history[2]
    a_prime = history[3]
    o_dprime = history[4]
    a_index = model.env.A_S.index(a)
    ap_index = model.env.A_S.index(a_prime)
    counts = np.zeros([len(model.beliefState),len(model.beliefState),len(model.beliefState)])
    totalCounts = 0
    for (m_idx,m) in enumerate(model.env.SDE_Set):
        for (mp_idx,mp) in enumerate(model.env.SDE_Set):
            for (mdp_idx,mdp) in enumerate(model.env.SDE_Set):
                # multFactor1 = int(mp[0] == o) #BUG: Collins pseudocode uses these masks. However, o and o' correspond to m and m' respectively, not m' and m".
                # multFactor2 = int(mdp[0] == o_prime)
                multFactor1 = model.beliefHistory[1][mp_idx]
                multFactor2 = model.beliefHistory[2][mdp_idx]
                counts[m_idx][mp_idx][mdp_idx] = multFactor1 * multFactor2 * (dirichlet.mean(model.TCounts[ap_index, mp_idx, :])[mdp_idx]) * (dirichlet.mean(model.TCounts[a_index, m_idx, :])[mp_idx]) * model.beliefHistory[0][m_idx]
                totalCounts = totalCounts + counts[m_idx][mp_idx][mdp_idx]

    if useBudd:
        max_rows = np.argwhere(np.array(model.beliefHistory[0]) == np.amax(np.array(model.beliefHistory[0])))
        max_rows2 = np.argwhere(np.array(model.beliefHistory[1]) == np.amax(np.array(model.beliefHistory[1])))
        if max_rows.size != 1 or max_rows2.size != 1:
            counts[:,:,:] = 0
        else:
            max_row = max_rows[0]
            max_row2 = max_rows2[0]
            counts[np.arange(len(model.env.SDE_Set)) != max_row, np.arange(len(model.env.SDE_Set)) != max_row2, :] = 0
            
    for m in range(len(model.env.SDE_Set)):
        for mp in range(len(model.env.SDE_Set)):
            for mdp in range(len(model.env.SDE_Set)):
                counts[m][mp][mdp] = counts[m][mp][mdp] / totalCounts
                model.OneTCounts[a_index][ap_index][m][mp][mdp] = model.OneTCounts[a_index][ap_index][m][mp][mdp] + counts[m][mp][mdp]
    #Note: Not necessary to do updateOneStepProbabilities (analagous to Algorithm 12) since this is handled by the dirichlet distributions
    

# Algorithm 18: sPOMDP Model State Splitting
def trySplit(model, revisedSplitting):
    G_ma = computeGains(model)
    G = []
    # Generate the list G that is used to order the model splitting
    mTrajLengths = [len(sde) for sde in model.env.SDE_Set]
    sortedIndexes = sorted(range(len(mTrajLengths)),key=mTrajLengths.__getitem__)
    for length in set(mTrajLengths):
        tripletStorage = []
        gainStorage = []
        for m in [i for i in sortedIndexes if mTrajLengths[i] == length]:
            # Get all of the relevant gains
            for a in range(len(model.env.A_S)):
                tripletStorage.append(((model.env.SDE_Set[m],model.env.A_S[a]),G_ma[a][m]))
                gainStorage.append(G_ma[a][m])
        sortedGainIndexes = sorted(range(len(gainStorage)),key=gainStorage.__getitem__)
        for index in sortedGainIndexes[::-1]: # Note: using the reverse of the list since sorted goes in ascending order of the gains
            G.append(tripletStorage[index])
        
    print("G")
    print(G)
    for gs in G:
        state = gs[0][0]
        action = gs[0][1]
        gainValue = gs[1]

        if revisedSplitting:
            firstOb = [sde[0] for sde in model.env.SDE_Set]
            skipGainPair = False
            for (obNum, ob) in enumerate(firstOb):
                if ob == state[0] and len(model.env.SDE_Set[obNum]) > 1: #Check the first observation of each SDE and, if it has a different first action than the variable "action", then skip as this would generate an invalid SDE
                    if model.env.SDE_Set[obNum][1] != action:
                        skipGainPair = True
                        break
            if skipGainPair:
                continue
        
        if gainValue > model.minGain:
            #Set m1 and m2 to be the two most likely states that are transitioned into from state m taking action a
            m_index = model.env.SDE_Set.index(state)
            a_index = model.env.A_S.index(action)
            transitionSetProbs = dirichlet.mean(model.TCounts[a_index, m_index, :])
            orderedVals = copy.deepcopy(transitionSetProbs)
            orderedVals.sort()
            prob1 = orderedVals[-1] #largest probability
            prob2 = orderedVals[-2] #second largest probability
            sde1_idx = np.where(transitionSetProbs == prob1)[0][0]
            if np.where(transitionSetProbs == prob1)[0].size > 1: # In this case, the most likely probability actually occurs twice (e.g. a 50-50 transition split)
                sde2_idx = np.where(transitionSetProbs == prob1)[0][1]
            else:
                sde2_idx = np.where(transitionSetProbs == prob2)[0][0]

            m1 = model.env.get_SDE()[sde1_idx]
            m2 = model.env.get_SDE()[sde2_idx]
            
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

                model.env.SDE_Set.append(newOutcome1)
                model.env.SDE_Set.append(newOutcome2)
                model.env.SDE_Set.remove(state)
                
                # Due to an issue in Collins pseudocode, the inserted SDEs are not guaranteed to correctly replace the older SDEs. Thus, if we are removing an SDE, it is best to rebuild the trie
                model.trieHead = TrieNode(None,[])
                for sde in model.env.SDE_Set:
                    insertSequence(model.trieHead, sde)
                
                print("Split the model. New States: ")
                print(model.env.SDE_Set)
                
                model.reinitializeModel()
                return True
            
            elif len(outcomesToAdd) == 1:
                model.env.SDE_Set.append(outcomesToAdd[0])
                print("Split the model. New States: ")
                print(model.env.SDE_Set)
                model.reinitializeModel()
                return True
            
    return False

# Compute gains according to equation 6.10 (Helper Function for Algorithm 18)
def computeGains(model):
    G = np.zeros([len(model.env.A_S),len(model.env.SDE_Set)])

    # Calculate some matrices that are used in calculating w_ma later
    mSinglePrimeSum_aPrime = np.sum(model.OneTCounts,axis = 4) #The total number of times the m' state is entered from state m under action a with respect to action a'
    mSinglePrimeSum = np.sum(mSinglePrimeSum_aPrime,axis = 0) #The total number of times the m' state is entered from state m under action a
    mPrimeSum = np.sum(np.sum(mSinglePrimeSum, axis = 0), axis=0) #The total number of times the m' state is entered

    for mp in range(len(model.env.SDE_Set)):
        for ap in range(len(model.env.A_S)):
            sum = 0
            w_masum = 0
            for m in range(len(model.env.SDE_Set)):
                for a in range(len(model.env.A_S)):
                    w_ma = mSinglePrimeSum[a, m, mp] / mPrimeSum[mp]
                    w_masum = w_masum + w_ma
                    sum = sum + (w_ma * entropy((dirichlet.mean(model.OneTCounts[a, ap, m, mp, :])), base=len(model.env.SDE_Set)))
            G[ap][mp] = entropy((dirichlet.mean(model.TCounts[ap, mp, :])), base=len(model.env.SDE_Set)) - sum
    return G

# Helper function to calculate the transition probabilities
def calcTransitionProbabilities(model):
    transProbs = np.zeros([len(model.env.A_S),len(model.env.SDE_Set),len(model.env.SDE_Set)])
    for a in range(len(model.env.A_S)):
        for m in range(len(model.env.SDE_Set)):
            transProbs[a][m][:] = np.array(dirichlet.mean(model.TCounts[a, m, :]))
    return(transProbs)


def haveControlPolicy(model, prevObservation, confidence_factor, performed_experiment, localization_threshold):
    np.seterr(all='warn')
    import warnings
    warnings.filterwarnings('error')

    #<<New Work: Controlling the agent while generating the trajectory. This allows the agent to prioritize performing transitions it has yet to confidently learn>>

    new_Full_Transition = []
    transitionProbs = calcTransitionProbabilities(model)

    # perform localization if unsure of where we are
    nonzero_values = np.count_nonzero(model.beliefState)
    if performed_experiment is True or (nonzero_values > 1 and entropy(model.beliefState, base=nonzero_values) > localization_threshold):
        Matching_SDE = model.env.get_SDE(prevObservation)
        Chosen_SDE = np.array(Matching_SDE[np.random.randint(low = 0, high = len(Matching_SDE))])
        Chosen_SDE_Actions = Chosen_SDE[np.arange(start=1, stop = len(Chosen_SDE), step= 2, dtype=int)]
        for action in Chosen_SDE_Actions:
            new_Full_Transition.append(action)

        if performed_experiment is True:
            # now need to get ourselves to a random state (in case there's latent states)
            # choose a random number of actions that could get us to any of our model states
            rand_actions = random.choices(model.env.A_S, k=max(1,len(model.env.SDE_Set) - 1))
            for action in rand_actions:
                new_Full_Transition.append(action)
            performed_experiment = False


    else:  # try to perform experiments so that we learn what we don't know

        # perform experiment if we're in a place where we can
        current_state = np.argmax(model.beliefState)
        for action_idx in range(len(model.env.A_S)):

            if np.sum(model.TCounts[action_idx, current_state])  / len(model.env.SDE_Set) < confidence_factor:
                action = model.env.A_S[action_idx]
                new_Full_Transition.append(action)
                performed_experiment = True
                break

        # if not in a state of interest, try to go to a state of interest
        if performed_experiment is False:
            path = getPathToExperiment(model, transitionProbs, current_state, confidence_factor, localization_threshold)
            if len(path) == 0:
                print("Error: We're already at an experiment state")
                return (new_Full_Transition, performed_experiment)
            else:
                new_Full_Transition.append(path[0])

    return (new_Full_Transition, performed_experiment)


# returns a path to a state where experimentation needs to be done
# returns empty list if already at state where experiment can be performed
# returns None if it has done experimentation for all reliably reachable nodes
def getPathToExperiment(model, transitionProbs, current_state, confidence_factor, localization_threshold):
    confidences = np.sum(model.TCounts, axis=2) / len(model.env.SDE_Set)
    # get the states of interest. Use the 1 index so we get states (as opposed to actions), and use unique so we don't repeat states that need to explore 2 or more actions
    states_of_interest = np.unique(np.array(np.where(confidences < confidence_factor))[1,:])
    if states_of_interest.size == 0:
        import pdb; pdb.set_trace()
        return None

    if current_state in states_of_interest:
        return []

    max_depth = len(model.env.SDE_Set)
    root = Node(str(current_state), reward=0, probability=1, actions=[])
    prev_lvl_nodes = [root]
    depth = 1
    bestNode = None
    while depth <= max_depth:
        added_nodes = []
        for node in prev_lvl_nodes:

            # that means we already performed experiment
            if node.reward != 0:
                continue

            m_idx = int(node.name)
            for action_idx in range(len(model.env.A_S)):
                a = model.env.A_S[action_idx]
                row = transitionProbs[action_idx, m_idx, :]

                new_actions = node.actions.copy()
                new_actions.append(a)
                new_probability = np.amax(row)*node.probability
                if confidences[action_idx, m_idx] >= confidence_factor:
                    nonzero_values = np.count_nonzero(row)
                    # if there's only one max and it's by a decent amount, set that to 1
                    if nonzero_values == 1 or (nonzero_values > 1 and entropy(row, base=nonzero_values) < localization_threshold):
                        if np.count_nonzero(row == np.amax(row)) == 1:  # make sure only one max trans prob
                            not_in_ancestors = True
                            for ancestor in node.ancestors:
                                if ancestor.name == str(np.argmax(row)):  # we've already been to this node for our path
                                    not_in_ancestors = False
                                    break
                            if not_in_ancestors is True:
                                added_nodes.append(Node(str(np.argmax(row)), parent=node, reward=0, probability=new_probability, actions=new_actions))
                else:  # need to do experimentation
                    reward = confidences[action_idx, m_idx] / (confidence_factor - 1)
                    reward = reward * node.probability
                    added_nodes.append(Node(str(np.argmax(row)), parent=node, reward=reward, probability=new_probability, actions=new_actions))
                    if bestNode is None or reward > bestNode.reward:
                        bestNode = added_nodes[-1]

        if len(added_nodes) == 0:
            break
        prev_lvl_nodes = added_nodes
        depth = depth + 1

    if bestNode is None:  # never found a place to do an experiment from
        return None
    else:
        return bestNode.actions[:-1]  # don't return the last action as it's the experiment


def drawGraph(model, root, bestNode):
    # for line in UniqueDotExporter(root):
    #     print(line)

    def nodeattrfunc(n):
        toReturn = ""
        roundedReward = round(n.reward,3)
        sde_str = "\n("
        for m_a in model.env.SDE_Set[int(n.name)]:
            if m_a == "square":
                sde_str = sde_str + "&#9633;,"
            elif m_a == "diamond":
                sde_str = sde_str + "&#9674;,"
            else:
                sde_str = sde_str + m_a + ","
        sde_str = sde_str[:-1] + ')'  # -1 to get rid of comma
        if n == bestNode:
            toReturn = toReturn + 'color=forestgreen, fontcolor=black, fontname="Times-Bold", '
        else:
            toReturn = toReturn + 'fontname="Times-Roman", '
        toReturn = toReturn + 'label="' + n.name + sde_str + '\nR=' + str(roundedReward) + '"'
        if n.is_leaf and n.reward != 0:
            toReturn = toReturn + ', style=dashed'
        return toReturn

    def edgeattrfunc(n, child):
        toReturn = ""
        if child == bestNode:
            toReturn = toReturn + 'color=forestgreen, fontcolor=black, fontname="Times-Bold"'
        toReturn = toReturn + 'label=" %s"' % (child.actions[-1])
        if child.is_leaf and child.reward != 0:
            toReturn = toReturn + ', style=dashed'
        return toReturn
    
    UniqueDotExporter(root, edgeattrfunc=edgeattrfunc, nodeattrfunc=nodeattrfunc).to_dotfile("Test2_graph.dot")
    cmd = ["dot", "-Tpng", "Test2_graph.dot", "-o", "Test2_graph.png"]
    check_call(cmd)

    img=mpimg.imread("Test2_graph.png")
    imgplot = plt.imshow(img)
    plt.show()

