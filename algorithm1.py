from abc import ABC, abstractmethod
import copy
import random
from scipy.stats import dirichlet
import numpy as np
import os


class POMDP():
    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text.  Instead of T(s, a, s') being  probability number for each
    state/action/state triplet, we instead have T(s, a) return a list of (p, s')
    pairs.  We also keep track of the possible states, terminal states, and
    actions for each state. [page 615]"""

    def __init__(self, S, O, A, T_ml, Omega_ml, alpha, epsilon):
        self.A = A
        self.O = O
        self.S = S
        self.T_ml = T_ml
        self.Omega_ml = Omega_ml
        self.alpha = alpha
        self.epsilon = epsilon

    def T(self, state, action):
        """Transition model.  From a state and an action, return a list
        of (result-state, probability) pairs."""
        transitionPairs = [(n, (1-self.alpha)/(len(self.S)-1)) for n in range(len(self.S))]
        transitionPairs[self.T_ml[state][action]] = (T_ml[state][action], self.alpha)
        return transitionPairs

    def Omega(self, state):
        """Observation model. For a given state, return a list of (observed state, probability)"""
        observationPairs = [(n, (1-self.epsilon)/(len(self.O)-1)) for n in range(len(self.O))]
        observationPairs[self.Omega_ml[state]] = (Omega_ml[state], self.epsilon)
        return observationPairs


    def actions(self, state):
        """Set of actions that can be performed in this state.  By default, a
        fixed list of actions. Override this
        method if you need to specialize by state."""
        return self.A

    # returns the probability distribution of what the state is if SDE m is successful, starting with startingDistribution in the POMDP environment
    def getSDEDistribution(self, m):

        # if not startingDistribution:
        #     stateDistribution = [(1/len(self.S)) for i in range(len(self.S))]
        # else:
        #     stateDistribution = startingDistribution
        finalStateDistribution = [0 for i in range(len(self.S))]

        for startingState in self.S:
            # print("************** looking at state " + str(startingState) + " ************************")

            # stateDistribution = [(1/len(self.S)) for i in range(len(self.S))]
            stateDistribution = [0 for i in range(len(self.S))]
            stateDistribution[startingState] = 1

            for s in self.S:
                # chance of seeing the first observation in m
                stateDistribution[s] = stateDistribution[s] * self.Omega(s)[m[0]][1]
                # print("self.Omega(s)[m[0]][1] " + str(self.Omega(s)[m[0]][1]))
            index = 1

            # total = sum(stateDistribution)
            # normalize the distribution
            # if total != 0:
            #     for s in self.S:
            #         stateDistribution[s] = stateDistribution[s] / total

            # print(stateDistribution) 
            newStateDistribution = copy.deepcopy(stateDistribution)
            # for each of the observation-action pairs in m
            while index < len(m):
                # create a zero list for the new state distributions
                newStateDistribution = [0 for i in range(len(self.S))]
                # for each of the states and transitions, find the chance of going to the new state and making the observation in m
                # note thate m[index] is the action and m[index+1] is the observation
                for s in self.S:
                    for transition in self.T(s, m[index]):
                        (newState, chance) = transition
                        # print("Transitioning from " + str(s) + " with action " + str(m[index]) + " to state " + str(newState) + " with a chance of " + str(chance))

                        newStateDistribution[newState] = newStateDistribution[newState] + (stateDistribution[s] * chance * self.Omega(newState)[m[index+1]][1])
                        # print("newStateDistribution " + str(newStateDistribution))
                        # print("----------------------------------")
                        # print("self.Omega(newState)[m[index+1]][1] " + str(self.Omega(newState)[m[index+1]][1]))
                for s in self.S:
                    stateDistribution[s] = newStateDistribution[s]


                index = index + 2

            finalStateDistribution[startingState] = sum(newStateDistribution)

        # # normalize the distribution
        # total = sum(stateDistribution)
        # for s in self.S:
        #     stateDistribution[s] = stateDistribution[s] / total

        # print("unnormalized final dist " + str(finalStateDistribution))
        # normalize the distribution
        total = sum(finalStateDistribution)
        for s in self.S:
            finalStateDistribution[s] = finalStateDistribution[s] / total

        return finalStateDistribution

    # P(m'| m,a) required for Algorithm 1, returns the probability of being in the same state dictated by m_prime vs when m and a are performed
    def getSDEConditionalProbability(self, m, a, m_prime):
        m_dist = self.getSDEDistribution(m)
        # print(m_dist)
        # create a zero list for the new state distributions
        newStateDistribution = [0 for i in range(len(self.S))]
        for s in self.S:
            for transition in self.T(s, a):
                        (newState, chance) = transition
                        newStateDistribution[newState] = newStateDistribution[newState] + (m_dist[s] * chance)

        # since it was given that m and a were successful, normalize this distribution
        # total = sum(newStateDistribution)
        # for s in self.S:
        #     newStateDistribution[s] = newStateDistribution[s] / total

        # print("m and action a distribution: " + str(newStateDistribution))
        # now with this distribution perform the experiment m_prime
        m_prime_dist = self.getSDEDistribution(m_prime)

        print("m_prime: " + str(m_prime))
        print("newStateDistribution: " + str(newStateDistribution))
        print("m_prime_dist: " + str(m_prime_dist))

        final_dist = [a*b for a,b in zip(newStateDistribution, m_prime_dist)]
        # print(final_dist)

        return sum(final_dist)

        # m_dist = self.getSDEDistribution(m)
        # print("m is " + str(m))
        # print("m_prime is " + str(m_prime))
        # print("a is " + str(a))
        # print("m_dist is " + str(m_dist))
        # newStateDistribution = [0 for i in range(len(self.S))]
        # for s in self.S:
        #     for transition in self.T(s, a):
        #         (newState, chance) = transition
        #         newStateDistribution[newState] = newStateDistribution[newState] + (m_dist[s] * chance)

        # print("newStateDistribution is " + str(newStateDistribution))
        # total = sum(newStateDistribution)
        # # normalize the distribution
        # for s in self.S:
        #     m_dist[s] = newStateDistribution[s] / total

        # print(sum(m_dist))

        # m_prime_dist = self.getSDEDistribution(m_prime)
        # print("m_prime_dist is " + str(m_prime_dist))

        # final_dist = [0 for i in range(len(self.S))]
        # for s in self.S:
        #     # TODO: This is WRONG!!!
        #     final_dist[s] = (m_dist[s] * m_prime_dist[s])

        # print("final_dist is " + str(final_dist))

        # return sum(final_dist)


    #Returns an observation from the current state, as dependent upon epsilon
    def getObservation(self, currentState):
        likelyObservation = self.Omega_ml[currentState]
        if random.random() < self.epsilon:
            #Successfully performed the most likely observation
            o = likelyObservation
        else:
            #Will randomly select one of the other possible observations
            observationList = list(copy.deepcopy(self.O))
            observationList.remove(likelyObservation)
            random.shuffle(observationList)
            o = observationList[0]
        return o

    #Returns the next state (as dependent upon the transition function and alpha), given the current state and an action 
    def getNextState(self, currentState, action):
        likelyTransition = self.T_ml[currentState][action]
        if random.random() < self.alpha:
            #Successfully performed the most likely transition
            nextState = likelyTransition
        else:
            #Will randomly select one of the other possible states
            statesList = list(copy.deepcopy(self.S))
            statesList.remove(likelyTransition)
            random.shuffle(statesList)
            nextState = statesList[0]
        return nextState

    #NOTE: This function has not been implemented as instead the mean is being used instead of the maximum
    #       The advantage to using the mean is that there is a built-in function. Additionally, for the
    #       desired/target distribution, the mean and mode should be rather similar
    #Finds and returns the location of the maximum value of the dirichlet distribution
    #The parameter alphas is a np.array
    #This location is returned as a numpy array
    # def getDirichletPointOfMax(alphas):
    #     stepSize = 0.01 #Due to float rounding errors, keep this to somewhat evenly divisible values, e.g. 0.1, 0.001, 0.01, etc.
    #     points = np.arange(0,1+stepSize,stepSize)

    #     maxLocation = []
    #     currentTestPoint = np.zeros((1,len(alphas[0])))
    #     iteratorValues = np.zeros((1,len(alphas[0])))

    #     while currentTestPoint[0][len(alphas[0] - 1)] <= 1.0:#as long as the last point hasn't been incremented past
    #         if sum(currentTestPoint == 1):
    #             dirichlet
    #         else:





    #Performs the SDE, returning the resulting trajectory t (composed of the actions taken and the observations recorded) and the resultingState.
    #explore is the probability ofr performing a random action instead of the action in
    #Note that the state returned is from the actual model state S, and not the SBL model M (that is being generated while learning) 
    def performSDE(self, startingState, SDE, explore):
        #iterate through the SDE's actions (which are odd indexes) and perform those actions, returning the sequence of actions and observations along the way
        #Note that SDE starts with an observation
        currentState = startingState
        o = self.getObservation(currentState)
        t = []
        #append the initial observation to the trajectory
        t.append(o)
        j = 1
        while j < len(SDE):
            a = SDE[j]

            #Perform a random action instead of the SDE action
            #NOTE: Paper is not explicit on whether a single random action should be taken or if the entire sequence of actions should be random
            #Currently, only one or more random actions may be selected, and a random action may be selected instead of the SDE action for each action
            if random.random() < explore:
                #Choose a random action, replacing the action a from the SDE
                actionList = list(copy.deepcopy(self.A))
                actionList.remove(SDE[j])
                random.shuffle(actionList)
                a = actionList[0]


            #Find and randomly choose a transition that originates from the currentState and takes action a
            currentState = self.getNextState(currentState,a)

            #Get observation from current state, then update t appropriately
            o = self.getObservation(currentState)

            t.append(a)
            t.append(o)

            j = j + 2

        return (t, currentState)

    # Algorithm 2. "numSDEs" is the number of SDEs to perform and "explore" is the probability of randomizing actions
    # modelStates is a list of SDEs that correspond to the model states
    # If desired, input the startingState where the experimentation begins (as the initial observation relies on this)
    def performExperiment(self, modelStates, numSDEs, explore, startingState = -1):
        if startingState < 0:
            #Pick a random startingState (needed to get the first observation)
            statesList = list(copy.deepcopy(self.S))
            random.shuffle(statesList)
            startingState = statesList[0]

        currentState = startingState

        fullT = []
        for i in range(1,numSDEs+1):
            o = self.getObservation(currentState)

            #Get model states m that have corresponding outcome sequences that begin with the current observation
            #NOTE: There should always be at least one model state that startes with the same observation, as one model state is created for each possible observation
            matching = []
            for sde in modelStates:
                if sde[0] == o:
                    #Same initial observation - add it to our list
                    matching.append(sde)

            random.shuffle(matching)
            randSDE = matching[0]

            # print("randSDE: " + str(randSDE))
            # print("starting state: " + str(currentState))

            (t, resultingState) = self.performSDE(currentState, randSDE, explore)
            # print("resulting state: " + str(resultingState))

            if i < numSDEs: 
                #a <- random action

                if i > (numSDEs*(3/4)) or random.random() < 0.25:
                    actionList = list(copy.deepcopy(self.A))
                    random.shuffle(actionList)
                    randAction = actionList[0]

                    #performAction()
                    newState = self.getNextState(resultingState, randAction)
                    currentState = newState

                    fullT = fullT + t + [randAction]
                else:
                    fullT = fullT + t[0:-1] 
                    #only add t[0:-1] since t start and ends with an observation, but we need the last item in t to be an action
                    currentState = resultingState
            else:
                fullT = fullT + t

        #learnTransitions()

        #Pre-process fullT so that when the agent performs an SDE, then the first observation is replaced 
        #by a label corresponding to that model state (NewLabel = modelStateNum + (numObservations))
        newTrajectory = copy.deepcopy(fullT)
        for sde in modelStates:
            modelStateNum = modelStates.index(sde)
            iteration = 0
            while iteration < (len(fullT)) - len(sde):
                if fullT[iteration:iteration+len(sde)] == sde:
                    #change the observation to a new model state
                    newTrajectory[iteration] = modelStateNum + len(self.O)
                iteration = iteration + 2
        fullT = newTrajectory

        beliefStates = np.ones((1,len(modelStates))) / len(modelStates) #Create a uniform distribution
        #Note: gammas is a 3 dimensional vector where the first dimension is m, second dimension is a, and third dimension is m'
        gammas = np.ones((len(modelStates), len(self.A), len(modelStates))) #create an array of hyperparamters initialized to ones

        iteration = 1 #Start at index 1 of fullT because the (a,o) pairs must be the observation o AFTER taking action a
        while iteration < len(fullT) - 1:
            a = fullT[iteration]
            processedO = fullT[iteration + 1] #o is the observation that occurs after taking action a

            o = processedO
            #See if we need to unprocess the observation (i.e. if it corresponds to an SDE)
            #Will need to get the first observation of that corresponding SDE
            if (processedO >= len(self.O)):
                sdeNum = processedO - len(self.O)
                o = modelStates[sdeNum][0]


            #Equation 3: Update gammas (Dirichlet hyperparameters)

            # #etta is assumed to be 1/(# of states that have the first observation o), i.e. 1/(# of times indicator function is 1)
            # #Thus, etta is only dependent upon m'
            # count = 0
            # for ms in modelStates:
            #     if ms[0] == o:
            #         count = count + 1
            # etta = 1.0/(count * len(modelStates))

            newGammas = copy.deepcopy(gammas)
            m_iter = 0

            #Need to iterate twice through each gamma value.
            #The first time (the loop below) is used to determine etta
            ettaSum = 0
            while m_iter < len(modelStates):
                    # a_iter = 0
                    a_iter = a
                    # while a_iter < len(self.A):
                    m_prime_iter = 0
                    while m_prime_iter < len(modelStates):
                        if processedO >= len(self.O):
                            #We have an SDE -> only update the model state that corresponds to that SDE number
                            if m_prime_iter == processedO - len(self.O):
                                ettaSum = ettaSum + (dirichlet.mean(gammas[m_iter][a_iter])[m_prime_iter] * beliefStates[0][m_iter])
                        elif modelStates[m_prime_iter][0] == o:#indicator function will be non-zero -> new gamma will be updated
                            # quantiles = np.zeros((1,beliefStates.size))
                            # #Evaluate the dirichlet using a one-hot vector representation at the dirichlet
                            # quantiles[0][m_prime_iter] = 1
                            ettaSum = ettaSum + (dirichlet.mean(gammas[m_iter][a_iter])[m_prime_iter] * beliefStates[0][m_iter])
                        m_prime_iter = m_prime_iter + 1
                    # a_iter = a_iter + 1
                    m_iter = m_iter + 1

            etta = 1/ettaSum
            m_iter = 0

            #The second loop through each gamma is used to update each gamma (now that the value of etta is known)
            while m_iter < len(modelStates):
                    # a_iter = 0
                    a_iter = a
                    # while a_iter < len(self.A):
                    m_prime_iter = 0
                    while m_prime_iter < len(modelStates):
                        if processedO >= len(self.O):
                            #We have an SDE -> only update the model state that corresponds to that SDE number
                            if m_prime_iter == processedO - len(self.O):
                                newGammas[m_iter][a_iter][m_prime_iter] = gammas[m_iter][a_iter][m_prime_iter] + (etta * dirichlet.mean(gammas[m_iter][a_iter])[m_prime_iter] * beliefStates[0][m_iter])    
                                # if (m_iter == 3) and (a_iter == 1):
                                    # print("iteration: " + str(iteration))
                                    # print("Processed Observation: " + str(processedO))
                                    # print("Target model state in the transition: " + str(m_prime_iter))
                                    # print("Belief State: " + str(beliefStates[0]))
                                    # print("Value added to gamma: " + str((etta * dirichlet.mean(gammas[m_iter][a_iter])[m_prime_iter] * beliefStates[0][m_iter])))
                                    # print("New Gammas: " + str(newGammas[m_iter][a_iter]))                            
                        elif modelStates[m_prime_iter][0] == o:#indicator function will be non-zero -> new gamma will be updated
                            # quantiles = np.zeros((1,beliefStates.size))
                            # #Evaluate the dirichlet using a one-hot vector representation at the dirichlet
                            # quantiles[0][m_prime_iter] = 1
                            # if (m_iter == 3) and (a_iter == 1):
                            #     print("Target model state in the transition: " + str(m_prime_iter))
                            #     print("Value added to gamma: " + str((etta * dirichlet.mean(gammas[m_iter][a_iter])[m_prime_iter] * beliefStates[0][m_iter])))
                            #     print("New Gammas (before the addition): " + str(newGammas[m_iter][a_iter]))
                            newGammas[m_iter][a_iter][m_prime_iter] = gammas[m_iter][a_iter][m_prime_iter] + (etta * dirichlet.mean(gammas[m_iter][a_iter])[m_prime_iter] * beliefStates[0][m_iter])
                        m_prime_iter = m_prime_iter + 1
                    # a_iter = a_iter + 1
                    m_iter = m_iter + 1

            gammas = copy.deepcopy(newGammas)

            #Equation 2 - needs to be processed after Equation 3 as the belief states use the transition probabilities T at time t (not t-1)
            newBeliefStates = np.zeros((1,beliefStates.size))
            
            #See if the agent successfully performed an SDE
            if (processedO > len(self.O)):
                #Agent performed an SDE successfully, so we will set the new belief state to reflect the successful SDE
                stateOfSDE = processedO - len(self.O)
                #Find number of belief states that match the current observation
                count = 0
                for ms in modelStates:
                    if ms[0] == modelStates[stateOfSDE][0]:
                        count = count + 1

                numNotMatching = len(modelStates) - count

                beta = (1 - self.alpha) / (len(modelStates) - 1)
                #update the non-SDE states that still have the same first observation
                for i in range(len(modelStates)):
                    if i == stateOfSDE:
                        newBeliefStates[0][i] = self.alpha + (beta * (numNotMatching/count))
                    elif modelStates[i][0] == modelStates[stateOfSDE][0]:
                        newBeliefStates[0][i] = beta + (beta * (numNotMatching/count))
            
            else:
                m_prime = 0

                while m_prime < beliefStates.size:
                    if modelStates[m_prime][0] == o: #indicator function will be non-zero -> new belief state will be non-zero
                        summation = 0
                        m_sigma = 0
                        while m_sigma < beliefStates.size:
                            # quantiles = np.zeros((1,beliefStates.size))
                            # #Evaluate the dirichlet using a one-hot vector representation at the dirichlet
                            # quantiles[0][sigma] = 1
                            summation = summation + (dirichlet.mean(gammas[m_sigma][a])[m_prime] * beliefStates[0][m_sigma])
                            m_sigma = m_sigma + 1
                        newBeliefStates[0][m_prime] = summation
                    m_prime = m_prime + 1

            #Need to normalize the belief states as the agent must be in one of the model states
            b_iter = 0
            summation = 0
            while b_iter < len(newBeliefStates[0]):
                summation = summation + newBeliefStates[0][b_iter]
                b_iter = b_iter + 1

            b_iter = 0
            while b_iter < len(newBeliefStates[0]):
                newBeliefStates[0][b_iter] = newBeliefStates[0][b_iter] / summation
                b_iter = b_iter + 1

            beliefStates = copy.deepcopy(newBeliefStates)

            iteration = iteration + 2
        print("fullT: " + str(fullT))
        print("final gammas: ")
        print(gammas)
        print("Transition Function for m = 0, a = x: " + str(dirichlet.mean(gammas[0][0])))
        print("Transition Function for m = 0, a = y: " + str(dirichlet.mean(gammas[0][1])))
        print("Transition Function for m = 1, a = x: " + str(dirichlet.mean(gammas[1][0])))
        print("Transition Function for m = 1, a = y: " + str(dirichlet.mean(gammas[1][1])))
        print("Transition Function for m = 2, a = x: " + str(dirichlet.mean(gammas[2][0])))
        print("Transition Function for m = 2, a = y: " + str(dirichlet.mean(gammas[2][1])))
        print("Transition Function for m = 3, a = x: " + str(dirichlet.mean(gammas[3][0])))
        print("Transition Function for m = 3, a = y: " + str(dirichlet.mean(gammas[3][1])))
        # print("Transition Function for m = state III, a = x: " + str(dirichlet.mean(gammas[2][0])))
        print("final beliefStates " + str(beliefStates))





def test1():
    # the set of observations
    O = range(2)
    # the set of actions
    A = range(2)
    #  the most likely state to transition to given T_ml[state][action]
    T_ml = [[1, 2], [1, 3], [0, 0], [2, 1]]
    #  the most likely observation given you're in that state
    Omega_ml = [0, 0, 1, 1]
    alpha = 1
    epsilon = 1
    E = POMDP(range(4), O, A, T_ml, Omega_ml, alpha, epsilon)

    # print(E.T(0, 0))
    # print(E.Omega(1))

    # M is the set of model states
    # S is a list of SDEs

    # modelStates is the set of model states, which are each an SDE that go o1, a1, o2, a2, o3...
    modelStates = []
    for observation in O:
        modelStates.append([observation])

    change = True
    while change is True:
        change = False
        modified = False
        # print("M is " + str(modelStates))
        for m in modelStates:
            if modified is True:
                break
            # print("m in modelStates")
            for a in A:
                if modified is True:
                    break
                # M_prime is the list of SDEs that extend off the original SDE m
                M_prime = copy.deepcopy(modelStates)
                # print("M_prime is " + str(M_prime))
                probabilities = [E.getSDEConditionalProbability(m, a, m_prime) for m_prime in M_prime]

                print("M_prime is " + str(M_prime))
                print("m is " + str(m))
                print("a is " + str(a))
                print("probabilities are " + str(probabilities))


                maxTrans = max(probabilities)
                if maxTrans < alpha:
                    print("maxTrans is " + str(maxTrans))
                    # TODO: make this quicker by not iterating through all M_prime for m_2_prime since it's redundant
                    m_1_prime = []
                    m_2_prime = []
                    for m_1_prime_candidate in M_prime:
                        for m_2_prime_candidate in M_prime:
                            if m_1_prime_candidate == m_2_prime_candidate:
                                continue
                            index = 0
                            while index < len(m_1_prime_candidate) and index < len(m_2_prime_candidate):
                                # check to see if they're different
                                if m_1_prime_candidate[index] != m_2_prime_candidate[index]:
                                    # if it's an action
                                    if index % 2 == 1:
                                        continue
                                    else:
                                        m_1_prime = m_1_prime_candidate
                                        m_2_prime = m_2_prime_candidate
                                        # print("m_1_prime " + str(m_1_prime))
                                        # print("m_2_prime " + str(m_2_prime))
                                        break  # end while
                                        break  # end first for
                                        break  # end second for
                                index = index + 1

                    # print("m_1_prime is " + str(m_1_prime))
                    # print("m_2_prime is " + str(m_2_prime))
                    m_1_new = [m[0], a]
                    m_1_new.extend(m_1_prime)
                    m_2_new = [m[0], a]
                    m_2_new.extend(m_2_prime)
                    print("m_1_new is " + str(m_1_new))
                    print("m_2_new is " + str(m_2_new))
                    # exit()
                    modelStates.append(m_1_new)
                    modelStates.append(m_2_new)
                    # TODO: append new SDE to the list of SDEs S
                    modelStates.remove(m)
                    change = True

                    if change is True:
                        modified = True
                        break
                        break

    print(modelStates)

def test2():
    print("Starting test 2")
    # the set of observations
    O = range(2)
    # the set of actions
    A = range(2)
    #  the most likely state to transition to given T_ml[state][action]
    T_ml = [[1, 2], [1, 3], [0, 0], [2, 1]]
    #  the most likely observation given you're in that state
    Omega_ml = [0, 0, 1, 1]
    alpha = 1
    epsilon = 1
    E = POMDP(range(4), O, A, T_ml, Omega_ml, alpha, epsilon)
    
    #Do test where model states are a perfect correlation to actual states (as determined in the paper)
    modelStates = [[0,1,1,0,0],[0,1,1,0,1],[1,0,0],[1,0,1]]
    # modelStates = [[0],[1]]
    # modelStates = [[0],[1,0,0],[1,0,1]]
    numSDEs = 1000
    explore = 0
    startingState = 3 #the starting state in the actual environment, not the model environment
    E.performExperiment(modelStates, numSDEs, explore, startingState)
    


test2()