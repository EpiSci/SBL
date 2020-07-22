import numpy as np

#This class represents a node and each node is a state in the model.
class sPOMDPNode():
    O_S = None
    A_S = None
    State_Size = None
    Alpha = None
    Epsilon = None
    def __init__(self, Observation, Action_Dictionary):
        self.Observation = Observation
        self.Action_Dictionary = Action_Dictionary #Action dictionary tells you what transition is equal to alpha for each action.
        
        self.Transition_Dictionary = {}
        for Action_Key in self.Action_Dictionary:
            Transition_Vector = np.zeros(self.State_Size)
            Transition_Vector[:] = (1-self.Alpha)/(self.State_Size-1)
            Transition_Vector[self.Action_Dictionary[Action_Key]] = self.Alpha
            self.Transition_Dictionary[Action_Key] = Transition_Vector
        
        self.Other_Observations = []
        for obs in self.O_S:
            if obs != self.Observation:
                self.Other_Observations.append(obs)

    def step_action(self, action:str):
        #try:
        return np.random.choice(np.arange(0, self.State_Size), p=self.Transition_Dictionary[action])
        #except:
        #    print("Invalid action.  Check your action inputs and action_dictionary.")
        #    exit()
            
    def get_observation(self):
        Random_Sample = np.random.random()
        if (Random_Sample<self.Epsilon):
            return self.Observation
        else:
            return self.Other_Observations[np.random.randint(low = 0, high = len(self.Other_Observations))]

#This class combines all of the nodes into a model.
class sPOMDPModelExample():
    def __init__(self):
        self.Node_Set = []
        self.SDE_Set = []

    def reset(self):
        #Select A Random Starting State
        self.Current_State = np.random.choice(self.Node_Set)
        return self.Current_State.get_observation()

    def step(self, action:str):
        self.Current_State = self.Node_Set[self.Current_State.step_action(action)]
        return self.Current_State.get_observation()

    def random_step(self):
        action = np.random.choice(self.A_S)
        return self.step(action), action

    def get_SDE(self, first_obs = None):
        if first_obs == None:
            return self.SDE_Set
        else:
            Matching_SDE = []
            for SDE in self.SDE_Set:
                if SDE[0] == first_obs:
                    Matching_SDE.append(SDE)
            return Matching_SDE

    def get_true_transition_probs(self):
        transitionProbs = np.zeros((len(self.A_S), len(self.Node_Set), len(self.Node_Set)))
        for (a_idx, a) in enumerate(self.A_S):
            for (s_idx, s) in enumerate(self.Node_Set):
                transitionProbs[a_idx, s_idx, :] = s.Transition_Dictionary[a]
        return transitionProbs

    def get_observation_probs(self):
        observationProbs = np.ones((len(self.O_S), len(self.Node_Set))) * ((1 - self.Epsilon) / (len(self.O_S) - 1))
        for (s_idx, s) in enumerate(self.Node_Set):
            o_idx = self.O_S.index(s.Observation)
            observationProbs[o_idx, s_idx] = self.Epsilon
        return observationProbs


#This class extends the generic sPOMDP model. This model is the one from figure 2.
class Example1(sPOMDPModelExample):
    def __init__(self):
        sPOMDPModelExample.__init__(self)
        #Set Environment Details
        self.O_S = ["square", "diamond"] #Observation Set
        self.A_S = ["x", "y"] #Action Set
        self.State_Size = 4
        self.Alpha = 0.99
        self.Epsilon = 0.99
        sPOMDPNode.O_S = self.O_S
        sPOMDPNode.A_S = self.A_S
        sPOMDPNode.State_Size = self.State_Size
        sPOMDPNode.Alpha = self.Alpha
        sPOMDPNode.Epsilon = self.Epsilon

        #Use Already Known SDE
        self.SDE_Set.append([self.O_S[0], self.A_S[1], self.O_S[1], self.A_S[0], self.O_S[0]])
        self.SDE_Set.append([self.O_S[0], self.A_S[1], self.O_S[1], self.A_S[0], self.O_S[1]])
        self.SDE_Set.append([self.O_S[1], self.A_S[0], self.O_S[0]])
        self.SDE_Set.append([self.O_S[1], self.A_S[0], self.O_S[1]])

        #Generate States
        self.Node_Set.append(sPOMDPNode(Observation = "square", Action_Dictionary = {"x": 1, "y": 2})) #state 0
        self.Node_Set.append(sPOMDPNode(Observation = "square", Action_Dictionary = {"x": 1, "y": 3})) #state 1
        self.Node_Set.append(sPOMDPNode(Observation = "diamond", Action_Dictionary = {"x": 0, "y": 0})) #state 2
        self.Node_Set.append(sPOMDPNode(Observation = "diamond", Action_Dictionary = {"x": 2, "y": 1})) #state 3


#This class extends the generic sPOMDP model. This model is the from the environment from Figure 2, but only with 2 known SDEs (square or diamond)
class Example2(sPOMDPModelExample):
    def __init__(self):
        sPOMDPModelExample.__init__(self)
        #Set Environment Details
        self.O_S = ["diamond", "square"] #Observation Set
        self.A_S = ["x", "y"] #Action Set
        self.State_Size = 4
        self.Alpha = 0.99
        self.Epsilon = 0.99
        sPOMDPNode.O_S = self.O_S
        sPOMDPNode.A_S = self.A_S
        sPOMDPNode.State_Size = self.State_Size
        sPOMDPNode.Alpha = self.Alpha
        sPOMDPNode.Epsilon = self.Epsilon

        #Use Already Known SDE
        self.SDE_Set.append([self.O_S[0]])
        self.SDE_Set.append([self.O_S[1]])

        #Generate States
        self.Node_Set.append(sPOMDPNode(Observation = "square", Action_Dictionary = {"x": 1, "y": 2})) #state 0
        self.Node_Set.append(sPOMDPNode(Observation = "square", Action_Dictionary = {"x": 1, "y": 3})) #state 1
        self.Node_Set.append(sPOMDPNode(Observation = "diamond", Action_Dictionary = {"x": 0, "y": 0})) #state 2
        self.Node_Set.append(sPOMDPNode(Observation = "diamond", Action_Dictionary = {"x": 2, "y": 1})) #state 3

#This class extends the generic sPOMDP model. This model is the from the environment from Figure 3, but only with 3 known SDEs (rose, volcano, or nothing)
class Example3(sPOMDPModelExample):
    def __init__(self):
        sPOMDPModelExample.__init__(self)
        #Set Environment Details
        self.O_S = ["rose", "volcano","nothing"] #Observation Set
        self.A_S = ["b", "f", "t"] #Action Set
        self.State_Size = 4
        self.Alpha = 0.99
        self.Epsilon = 0.99
        sPOMDPNode.O_S = self.O_S
        sPOMDPNode.A_S = self.A_S
        sPOMDPNode.State_Size = self.State_Size
        sPOMDPNode.Alpha = self.Alpha
        sPOMDPNode.Epsilon = self.Epsilon

        #Use Already Known SDE
        self.SDE_Set.append([self.O_S[0]])
        self.SDE_Set.append([self.O_S[1]])
        self.SDE_Set.append([self.O_S[2]])

        #Generate States
        self.Node_Set.append(sPOMDPNode(Observation = "rose", Action_Dictionary = {"f": 3, "b": 2, "t": 0})) #state 0
        self.Node_Set.append(sPOMDPNode(Observation = "volcano", Action_Dictionary = {"f": 2, "b": 3, "t": 1})) #state 1
        self.Node_Set.append(sPOMDPNode(Observation = "nothing", Action_Dictionary = {"f": 0, "b": 1, "t": 3})) #state 2
        self.Node_Set.append(sPOMDPNode(Observation = "nothing", Action_Dictionary = {"f": 1, "b": 0, "t": 2})) #state 3


#This class extends the generic sPOMDP model. This model is the from the environment from Figure 3, but only with 3 known SDEs (rose, volcano, or nothing)
class Example32(sPOMDPModelExample):
    def __init__(self):
        sPOMDPModelExample.__init__(self)
        #Set Environment Details
        self.O_S = ["rose", "volcano","nothing"] #Observation Set
        self.A_S = ["b", "f", "t"] #Action Set
        self.State_Size = 4
        self.Alpha = 0.99
        self.Epsilon = 0.99
        sPOMDPNode.O_S = self.O_S
        sPOMDPNode.A_S = self.A_S
        sPOMDPNode.State_Size = self.State_Size
        sPOMDPNode.Alpha = self.Alpha
        sPOMDPNode.Epsilon = self.Epsilon

        #Use Already Known SDE
        self.SDE_Set.append([self.O_S[0]])
        self.SDE_Set.append([self.O_S[1]])
        self.SDE_Set.append([self.O_S[2], self.A_S[0], self.O_S[1]])
        self.SDE_Set.append([self.O_S[2], self.A_S[0], self.O_S[0]])

        #Generate States
        self.Node_Set.append(sPOMDPNode(Observation = "rose", Action_Dictionary = {"f": 3, "b": 2, "t": 0})) #state 0
        self.Node_Set.append(sPOMDPNode(Observation = "volcano", Action_Dictionary = {"f": 2, "b": 3, "t": 1})) #state 1
        self.Node_Set.append(sPOMDPNode(Observation = "nothing", Action_Dictionary = {"f": 0, "b": 1, "t": 3})) #state 2
        self.Node_Set.append(sPOMDPNode(Observation = "nothing", Action_Dictionary = {"f": 1, "b": 0, "t": 2})) #state 3


#This class extends the generic sPOMDP model. This model is the from the environment from Figure 4, but only with 2 known SDEs (goal, nothing)
class Example4(sPOMDPModelExample):
    def __init__(self):
        sPOMDPModelExample.__init__(self)
        #Set Environment Details
        self.O_S = ["goal","nothing"] #Observation Set
        self.A_S = ["east", "west"] #Action Set
        self.State_Size = 4
        self.Alpha = 0.99
        self.Epsilon = 0.99
        sPOMDPNode.O_S = self.O_S
        sPOMDPNode.A_S = self.A_S
        sPOMDPNode.State_Size = self.State_Size
        sPOMDPNode.Alpha = self.Alpha
        sPOMDPNode.Epsilon = self.Epsilon

        #Use Already Known SDE
        self.SDE_Set.append([self.O_S[0]])
        self.SDE_Set.append([self.O_S[1]])

        #Generate States
        self.Node_Set.append(sPOMDPNode(Observation = "goal", Action_Dictionary = {"east": 1, "west": 3})) #state goal
        self.Node_Set.append(sPOMDPNode(Observation = "nothing", Action_Dictionary = {"east": 2, "west": 0})) #state left
        self.Node_Set.append(sPOMDPNode(Observation = "nothing", Action_Dictionary = {"east": 3, "west": 1})) #state middle
        self.Node_Set.append(sPOMDPNode(Observation = "nothing", Action_Dictionary = {"east": 0, "west": 2})) #state right

class Example42(sPOMDPModelExample):
    def __init__(self):
        sPOMDPModelExample.__init__(self)
        #Set Environment Details
        self.O_S = ["goal","nothing"] #Observation Set
        self.A_S = ["east", "west"] #Action Set
        self.State_Size = 4
        self.Alpha = 0.99
        self.Epsilon = 0.99
        sPOMDPNode.O_S = self.O_S
        sPOMDPNode.A_S = self.A_S
        sPOMDPNode.State_Size = self.State_Size
        sPOMDPNode.Alpha = self.Alpha
        sPOMDPNode.Epsilon = self.Epsilon

        #Use Already Known SDE
        self.SDE_Set.append([self.O_S[0]])
        self.SDE_Set.append([self.O_S[1], self.A_S[0], self.O_S[1], self.A_S[0], self.O_S[1]])
        self.SDE_Set.append([self.O_S[1], self.A_S[0], self.O_S[1], self.A_S[0], self.O_S[0]])
        self.SDE_Set.append([self.O_S[1], self.A_S[0], self.O_S[0]])


        #Generate States
        self.Node_Set.append(sPOMDPNode(Observation = "goal", Action_Dictionary = {"east": 1, "west": 3})) #state goal
        self.Node_Set.append(sPOMDPNode(Observation = "nothing", Action_Dictionary = {"east": 2, "west": 0})) #state left
        self.Node_Set.append(sPOMDPNode(Observation = "nothing", Action_Dictionary = {"east": 3, "west": 1})) #state middle
        self.Node_Set.append(sPOMDPNode(Observation = "nothing", Action_Dictionary = {"east": 0, "west": 2})) #state right


#This class extends the generic sPOMDP model. This model is the from the environment from Figure 5, 
#but only with 5 known starting SDEs (nothing, LRVForward, MRVForward, LRVDocked, MRVDocked)
class Example5(sPOMDPModelExample):
    def __init__(self):
        sPOMDPModelExample.__init__(self)
        #Set Environment Details
        self.O_S = ["nothing","LRVForward", "MRVForward", "LRVDocked", "MRVDocked"] #Observation Set
        self.A_S = ["b", "f", "t"] #Action Set
        self.State_Size = 8
        self.Alpha = 0.99
        self.Epsilon = 0.99
        sPOMDPNode.O_S = self.O_S
        sPOMDPNode.A_S = self.A_S
        sPOMDPNode.State_Size = self.State_Size
        sPOMDPNode.Alpha = self.Alpha
        sPOMDPNode.Epsilon = self.Epsilon

        #Use Already Known SDE
        self.SDE_Set.append([self.O_S[0]])
        self.SDE_Set.append([self.O_S[1]])
        self.SDE_Set.append([self.O_S[2]])
        self.SDE_Set.append([self.O_S[3]])        
        self.SDE_Set.append([self.O_S[4]])

        #Generate States
        self.Node_Set.append(sPOMDPNode(Observation = "LRVDocked", Action_Dictionary = {"b": 7, "f": 4, "t": 1})) #state 0
        self.Node_Set.append(sPOMDPNode(Observation = "MRVForward", Action_Dictionary = {"b": 1, "f": 1, "t": 4})) #state 1
        self.Node_Set.append(sPOMDPNode(Observation = "MRVForward", Action_Dictionary = {"b": 3, "f": 1, "t": 5})) #state 2
        self.Node_Set.append(sPOMDPNode(Observation = "nothing", Action_Dictionary = {"b": 0, "f": 2, "t": 6})) #state 3
        self.Node_Set.append(sPOMDPNode(Observation = "nothing", Action_Dictionary = {"b": 7, "f": 5, "t": 1})) #state 4
        self.Node_Set.append(sPOMDPNode(Observation = "LRVForward", Action_Dictionary = {"b": 4, "f": 6, "t": 2})) #state 5
        self.Node_Set.append(sPOMDPNode(Observation = "LRVForward", Action_Dictionary = {"b": 6, "f": 6, "t": 3})) #state 6
        self.Node_Set.append(sPOMDPNode(Observation = "MRVDocked", Action_Dictionary = {"b": 7, "f": 4, "t": 1})) #state 7


# The ladder environment. Length is the number of intermediate diamond states between the squares. y actions lead forward, x falls off
class Example6(sPOMDPModelExample):
    def __init__(self):
        sPOMDPModelExample.__init__(self)
        #Set Environment Details
        self.O_S = ["square", "diamond"] #Observation Set
        self.A_S = ["x", "y"] #Action Set
        self.State_Size = 4
        self.Alpha = 0.99
        self.Epsilon = 0.99
        sPOMDPNode.O_S = self.O_S
        sPOMDPNode.A_S = self.A_S
        sPOMDPNode.State_Size = self.State_Size
        sPOMDPNode.Alpha = self.Alpha
        sPOMDPNode.Epsilon = self.Epsilon

        #Use Already Known SDE
        self.SDE_Set.append([self.O_S[0], self.A_S[1], self.O_S[1]])
        self.SDE_Set.append([self.O_S[1], self.A_S[1], self.O_S[1]])
        self.SDE_Set.append([self.O_S[1], self.A_S[1], self.O_S[0]])
        self.SDE_Set.append([self.O_S[0], self.A_S[1], self.O_S[0]])

        #Generate States
        self.Node_Set.append(sPOMDPNode(Observation = "square", Action_Dictionary = {"x": 0, "y": 1})) #state 0
        self.Node_Set.append(sPOMDPNode(Observation = "diamond", Action_Dictionary = {"x": 0, "y": 2})) #state 1
        self.Node_Set.append(sPOMDPNode(Observation = "diamond", Action_Dictionary = {"x": 0, "y": 3})) #state 2
        self.Node_Set.append(sPOMDPNode(Observation = "square", Action_Dictionary = {"x": 0, "y": 3})) #state 3


# The ladder environment without SDEs. y actions lead forward, x falls off
class Example7(sPOMDPModelExample):
    def __init__(self):
        sPOMDPModelExample.__init__(self)
        #Set Environment Details
        self.O_S = ["square", "diamond"] #Observation Set
        self.A_S = ["x", "y"] #Action Set
        self.State_Size = 4
        self.Alpha = 0.99
        self.Epsilon = 0.99
        sPOMDPNode.O_S = self.O_S
        sPOMDPNode.A_S = self.A_S
        sPOMDPNode.State_Size = self.State_Size
        sPOMDPNode.Alpha = self.Alpha
        sPOMDPNode.Epsilon = self.Epsilon

        #Use Already Known SDE
        self.SDE_Set.append([self.O_S[0]])
        self.SDE_Set.append([self.O_S[1]])

        #Generate States
        self.Node_Set.append(sPOMDPNode(Observation = "square", Action_Dictionary = {"x": 0, "y": 1})) #state 0
        self.Node_Set.append(sPOMDPNode(Observation = "diamond", Action_Dictionary = {"x": 0, "y": 2})) #state 1
        self.Node_Set.append(sPOMDPNode(Observation = "diamond", Action_Dictionary = {"x": 0, "y": 3})) #state 2
        self.Node_Set.append(sPOMDPNode(Observation = "square", Action_Dictionary = {"x": 0, "y": 3})) #state 3


# The slide environment without SDEs
class Example8(sPOMDPModelExample):
    def __init__(self):
        sPOMDPModelExample.__init__(self)
        #Set Environment Details
        self.O_S = ["square", "diamond"] #Observation Set
        self.A_S = ["x", "y"] #Action Set
        self.State_Size = 4
        self.Alpha = 0.99
        self.Epsilon = 0.99
        sPOMDPNode.O_S = self.O_S
        sPOMDPNode.A_S = self.A_S
        sPOMDPNode.State_Size = self.State_Size
        sPOMDPNode.Alpha = self.Alpha
        sPOMDPNode.Epsilon = self.Epsilon

        #Use Already Known SDE
        self.SDE_Set.append([self.O_S[0]])
        self.SDE_Set.append([self.O_S[1]])

        #Generate States
        self.Node_Set.append(sPOMDPNode(Observation = "square", Action_Dictionary = {"x": 0, "y": 1})) #state 0
        self.Node_Set.append(sPOMDPNode(Observation = "diamond", Action_Dictionary = {"x": 2, "y": 2})) #state 1
        self.Node_Set.append(sPOMDPNode(Observation = "diamond", Action_Dictionary = {"x": 3, "y": 3})) #state 2
        self.Node_Set.append(sPOMDPNode(Observation = "square", Action_Dictionary = {"x": 0, "y": 2})) #state 3

# The slide environment with SDEs
class Example9(sPOMDPModelExample):
    def __init__(self):
        sPOMDPModelExample.__init__(self)
        #Set Environment Details
        self.O_S = ["square", "diamond"] #Observation Set
        self.A_S = ["x", "y"] #Action Set
        self.State_Size = 4
        self.Alpha = 0.99
        self.Epsilon = 0.99
        sPOMDPNode.O_S = self.O_S
        sPOMDPNode.A_S = self.A_S
        sPOMDPNode.State_Size = self.State_Size
        sPOMDPNode.Alpha = self.Alpha
        sPOMDPNode.Epsilon = self.Epsilon

        #Use Already Known SDE
        self.SDE_Set.append([self.O_S[0], self.A_S[1], self.O_S[1], self.A_S[0], self.O_S[1]])
        self.SDE_Set.append([self.O_S[1], self.A_S[0], self.O_S[1]])
        self.SDE_Set.append([self.O_S[1], self.A_S[0], self.O_S[0]])
        self.SDE_Set.append([self.O_S[0], self.A_S[1], self.O_S[1], self.A_S[0], self.O_S[0]])

        #Generate States
        self.Node_Set.append(sPOMDPNode(Observation = "square", Action_Dictionary = {"x": 0, "y": 1})) #state 0
        self.Node_Set.append(sPOMDPNode(Observation = "diamond", Action_Dictionary = {"x": 2, "y": 2})) #state 1
        self.Node_Set.append(sPOMDPNode(Observation = "diamond", Action_Dictionary = {"x": 3, "y": 3})) #state 2
        self.Node_Set.append(sPOMDPNode(Observation = "square", Action_Dictionary = {"x": 0, "y": 2})) #state 3

#Used in Algorithm 3 code as a generic model.
class genericModel(sPOMDPModelExample):
    def __init__(self, observationSet,actionSet, stateSize, SDE_Set, alpha, epsilon, environmentNodes):
        sPOMDPModelExample.__init__(self)
        #Set Environment Details
        self.O_S = observationSet
        self.A_S = actionSet
        self.State_Size = stateSize
        self.Alpha = alpha
        self.Epsilon = epsilon
        sPOMDPNode.O_S = self.O_S
        sPOMDPNode.A_S = self.A_S
        sPOMDPNode.State_Size = self.State_Size
        sPOMDPNode.Alpha = self.Alpha
        sPOMDPNode.Epsilon = self.Epsilon

        self.SDE_Set = SDE_Set

        #Generate States
        self.Node_Set = environmentNodes



# Calculate the gain of an environment given the transition probabilities and the one-step extension gammas
# Returns the associated entropy of each (m,a) pair and the associated gain
def calculateGain(env, Action_Probs, OneStep_Gammas):
    entropyMA = np.zeros((len(env.A_S),len(env.SDE_Set))) #index as action, model number 
    gainMA = np.zeros((len(env.A_S),len(env.SDE_Set))) #index as action, model number

    oneStep_TransitionProbs = OneStep_Gammas / np.reshape(np.repeat(np.sum(OneStep_Gammas, axis = 4),len(env.SDE_Set),axis=3),OneStep_Gammas.shape)
    mSinglePrimeSum_aPrime = np.sum(OneStep_Gammas,axis = 4) #The total number of times the m' state is entered from state m under action a with respect to action a'
    mSinglePrimeSum = np.sum(mSinglePrimeSum_aPrime,axis = 0) #The total number of times the m' state is entered from state m under action a
    # mPrimeSum = np.sum(Action_Gammas, axis = 2) #The total number of times the action a is executed from state m
    mPrimeSum = np.sum(np.sum(mSinglePrimeSum, axis = 0), axis=0) #The total number of times the m' state is entered

    # Calculate the transition entropies H(Ttma). Calculate the gain values using the OneStep_Gammas
    for mPrime_idx, mPrime in enumerate(env.SDE_Set):
        for aPrime_idx, aPrime in enumerate(env.A_S):
            transitionSetProbs = Action_Probs[aPrime_idx,mPrime_idx,:]
            transitionSetEntropy = np.sum(np.multiply(transitionSetProbs,(np.log(transitionSetProbs) / np.log(len(env.SDE_Set))))) * -1
            entropyMA[aPrime_idx,mPrime_idx] = transitionSetEntropy

            sigma = 0
            w_maSum = 0
            for a_idx, a in enumerate(env.A_S):
                for m_idx, m in enumerate(env.SDE_Set):

                    w_ma = mSinglePrimeSum[a_idx, m_idx, mPrime_idx] / mPrimeSum[mPrime_idx]
                    w_maSum = w_maSum + w_ma
                    # oneStepTransitionProb = oneStep_TransitionProbs[aPrime_idx,a_idx,m_idx,mPrime_idx,:]
                    oneStepTransitionProb = oneStep_TransitionProbs[a_idx,aPrime_idx,m_idx,mPrime_idx,:]
                    oneStep_TransitionEntropy = np.sum(np.multiply(oneStepTransitionProb,(np.log(oneStepTransitionProb) / np.log(len(env.SDE_Set))))) * -1
                    sigma = (w_ma * oneStep_TransitionEntropy) + sigma

            gainMA[aPrime_idx,mPrime_idx] = entropyMA[aPrime_idx,mPrime_idx] - sigma


        printOneStep = False
    if printOneStep:
        import csv
        c = csv.writer(open("TestingFigure5April29Trial1.csv", "w"))
        
        c.writerow(["entropyMA"])
        c.writerow(entropyMA)

        c.writerow(["Gain: "])
        c.writerow(gainMA) 

        c.writerow(["Action_Probs: "])
        for a1_idx, a1 in enumerate(env.A_S):
            for m1_idx, m1 in enumerate(env.SDE_Set):
                c.writerow(Action_Probs[a1_idx, m1_idx, :])

        for a1_idx, a1 in enumerate(env.A_S):
            for a2_idx, a2 in enumerate(env.A_S):    
                for m1_idx, m1 in enumerate(env.SDE_Set):
                    for m2_idx, m2 in enumerate(env.SDE_Set):
                        c.writerow(["One-Step Transition Gamma: " + str(m1) + " " + str(a1) + " " + str(m2) + " " +  str(a2) + " X"])
                        c.writerow(OneStep_Gammas[a1_idx, a2_idx, m1_idx, m2_idx, :])
                        w_ma = mSinglePrimeSum[a_idx, m_idx, mPrime_idx] / mPrimeSum[mPrime_idx]
                        c.writerow(["Weight value that the transition m = " + str(m1) + " and a =  " + str(a1) + "causes the transition into m' = " + str(m2) + ":"])
                        c.writerow([str(mSinglePrimeSum[a1_idx, m1_idx, m2_idx] / mPrimeSum[m2_idx])])
                        # print("One-Step Transition Gamma: " + str(m1) + " " + str(a1) + " " + str(m2) + " " +  str(a1) + " X")
                        # print(OneStep_Gammas[a1_idx, a2_idx, m1_idx, m2_idx, :])



    # print(oneStep_TransitionProbs)
    printOneStepTransitions = False
    if printOneStepTransitions:
        print("One Step Transition Probs: ")
        for a1_idx, a1 in enumerate(env.A_S):
            for a2_idx, a2 in enumerate(env.A_S):    
                for m1_idx, m1 in enumerate(env.SDE_Set):
                    for m2_idx, m2 in enumerate(env.SDE_Set):
                        print("One-Step Transition Probs: " + str(m1) + " " + str(a1) + " " + str(m2) + " " +  str(a2) + " X")
                        print(OneStep_TransitoinProbs[a1_idx, a2_idx, m1_idx, m2_idx, :])

    print("entropyMA")
    print(entropyMA)

    print("Gain: ")
    print(gainMA)
    
    print("Transition Probabilities: ")
    print(Action_Probs)

    return(gainMA, entropyMA)

# Calculate the error of a model as defined in Equation 6.4 (pg 122) of Collins' Thesis
# env holds the SDEs for the model
def calculateError(env, modelTransitionProbs, T, gammas):
    doRelative = False

    Current_Observation = env.reset()
    SDE_List = env.get_SDE()
    state_List = env.Node_Set

    first_Observations_mod = [item[0] for item in SDE_List]
    first_Observations_env = [item.Observation for item in state_List]

    # Generate the transition probabilities for the environment
    envTransitionProbs = env.get_true_transition_probs()

    # Generate trajectory using environment
    Full_Transition = [Current_Observation]
    for num in range(0,T):
        Current_Observation, random_action = env.random_step()
        Full_Transition.append(random_action)
        Full_Transition.append(Current_Observation)

    # print("Full_Transition")
    # print(Full_Transition)

    # Generate a belief mask for each model state that indicates what the likelihood is of being in each model state given an observation
    Obs_Belief_Mask_mod = np.zeros((len(env.O_S), len(SDE_List)))
    for (o_idx, o) in enumerate(env.O_S):
        SDE_Chance = np.zeros(len(SDE_List))
        #Figure out how many SDEs correspond to the observation
        num_Correspond = first_Observations_mod.count(o)
        #Set the corresponding SDEs to 1 divided by that value
        SDE_Chance[(np.array(first_Observations_mod) == o)] = 1/num_Correspond
        Obs_Belief_Mask_mod[o_idx,:] = SDE_Chance

    # Generate a belief mask for each env state that indicates what the likelihood is of being in each env state given an observation
    Obs_Belief_Mask_env = np.zeros((len(env.O_S), len(state_List)))
    for (o_idx, o) in enumerate(env.O_S):
        Obs_Chance = np.zeros(len(state_List))
        #Figure out how many states correspond to the observation
        num_Correspond = first_Observations_env.count(o)
        #Set the corresponding states to 1 divided by that value
        Obs_Chance[(np.array(first_Observations_env) == o)] = 1/num_Correspond
        Obs_Belief_Mask_env[o_idx,:] = Obs_Chance

    # Generate P(o|m) matrix for model and environment
    Obs_Probs_mod = np.ones((len(env.O_S), len(SDE_List))) * ((1 - env.Epsilon) / ((len(env.O_S)) - 1))
    for (sde_idx, sde) in enumerate(SDE_List):
        o_idx = env.O_S.index(sde[0])
        Obs_Probs_mod[o_idx, sde_idx] = env.Epsilon
    Obs_Probs_env = env.get_observation_probs()

    # print("Obs_Probs_mod")
    # print(Obs_Probs_mod)
    # print("Obs_Probs_env")
    # print(Obs_Probs_env)

    # Generate starting belief states for environment and model using first observation
    Observation = Full_Transition[0]
    Observation_Idx = env.O_S.index(Observation)
    Belief_State_mod = Obs_Belief_Mask_mod[Observation_Idx].copy()
    Belief_State_env = Obs_Belief_Mask_env[Observation_Idx].copy()

    # determine the weights for the error from the confidence
    sum_of_row = np.sum(gammas, axis=2)
    weights = len(SDE_List) * np.ones((len(env.A_S), len(SDE_List)))
    weights = np.divide(weights, sum_of_row)
    weights = 1 - weights

    error = 0
    Transition_Idx = 0
    prev_error = 0
    while Transition_Idx < len(Full_Transition)//2:

        # print("Observation: " + str(Observation))
        # print("Belief_State_mod: " + str(Belief_State_mod))
        # print("Belief_State_env: " + str(Belief_State_env))

        # update the belief states with the new action, observation pair
        Observation = Full_Transition[Transition_Idx*2+2]
        Observation_Idx = env.O_S.index(Observation)
        Action = Full_Transition[Transition_Idx*2+1]
        Belief_Mask_mod = Obs_Belief_Mask_mod[Observation_Idx]
        Belief_Mask_env = Obs_Belief_Mask_env[Observation_Idx]

        Model_Action_Idx = env.A_S.index(Action)
        Belief_State_mod = np.dot(Belief_State_mod, modelTransitionProbs[Model_Action_Idx,:,:])
        Belief_State_env = np.dot(Belief_State_env, envTransitionProbs[Model_Action_Idx,:,:])

        # Belief_State_mod = Belief_State_mod*Belief_Mask_mod
        Belief_State_mod = Belief_State_mod/np.sum(Belief_State_mod)
        # Belief_State_env = Belief_State_env*Belief_Mask_env
        Belief_State_env = Belief_State_env/np.sum(Belief_State_env)

        # Compute error for the current belief states
        weight_vector = weights[env.A_S.index(Action), :]
        scale = np.dot(Belief_State_mod, weight_vector)
        error_vector = np.dot(Obs_Probs_mod, Belief_State_mod) - np.dot(Obs_Probs_env, Belief_State_env)
        
        if doRelative is True:
            error = error + (scale * np.sqrt(error_vector.dot(error_vector)))
        else:
            error = error + np.sqrt(error_vector.dot(error_vector))

        Belief_State_mod = Belief_State_mod*Belief_Mask_mod
        Belief_State_mod = Belief_State_mod/np.sum(Belief_State_mod)
        Belief_State_env = Belief_State_env*Belief_Mask_env
        Belief_State_env = Belief_State_env/np.sum(Belief_State_env)

        # if np.sqrt(error_vector.dot(error_vector)) >= prev_error:
            # print("Obs_Belief_Mask_mod")
            # print(Obs_Belief_Mask_mod)
            # print("Obs_Belief_Mask_env")
            # print(Obs_Belief_Mask_env)
            # print("Belief_State_mod")
            # print(Belief_State_mod)
            # print("Obs_Probs_mod")
            # print(Obs_Probs_mod)
            # print("Belief_State_env")
            # print(Belief_State_env)
            # print("Obs_Probs_env")
            # print(Obs_Probs_env)
            # print("np.dot(Obs_Probs_mod, Belief_State_mod)")
            # print(np.dot(Obs_Probs_mod, Belief_State_mod))
            # print("np.dot(Obs_Probs_env, Belief_State_env)")
            # print(np.dot(Obs_Probs_env, Belief_State_env))
            # print("error_vector")
            # print(error_vector)
            # print("np.sqrt(error_vector.dot(error_vector))")
            # print(np.sqrt(error_vector.dot(error_vector)))
            # print("---------------------")

        prev_error = np.sqrt(error_vector.dot(error_vector))

        Transition_Idx = Transition_Idx + 1

    return error / T

# calculates the absolute error. Error=1 when incorrect SDEs or # of SDEs, otherwise it's the average difference of each transition / 2
# note: it is assummed that env.Node_Set contains a minimum representation of the true environment nodes
def calculateAbsoluteError(env, modelTransitionProbs):
    num_of_env_states = len(env.Node_Set)
    if num_of_env_states != len(env.SDE_Set):
        return 1

    envTransitionProbs = env.get_true_transition_probs()
    # Generate P(o|m) matrix for environment
    Obs_Probs_env = env.get_observation_probs()

    # check that each SDE corresponds to one environment state
    SDEToNode = []
    for SDE in env.SDE_Set:
        probabilities = np.zeros(len(env.Node_Set))
        for env_state_num in range(len(env.Node_Set)):
            Belief_State = np.zeros(len(env.Node_Set))
            Belief_State[env_state_num] = 1
            first_obs_index = env.O_S.index(SDE[0])
            Belief_State = np.multiply(Obs_Probs_env[first_obs_index,:], Belief_State)

            Transition_Idx = 0
            while Transition_Idx < len(SDE)//2:

                Observation = SDE[Transition_Idx*2+2]
                Observation_Idx = env.O_S.index(Observation)
                Action = SDE[Transition_Idx*2+1]
                Action_Idx = env.A_S.index(Action)
                
                Belief_State = np.dot(Belief_State, envTransitionProbs[Action_Idx,:,:])
                Belief_State = np.multiply(Obs_Probs_env[Observation_Idx,:], Belief_State)

                Transition_Idx = Transition_Idx + 1

            probabilities[env_state_num] = np.sum(Belief_State)

        SDEToNode.append(probabilities.argmax())

    if len(SDEToNode) != len(set(SDEToNode)):
        return 1



    # SDEs are valid, so now calculate the absolute difference per transition / 2
    SDEToNode = np.array(SDEToNode)
    error = 0
    for a_idx in range(len(env.A_S)):
        permutatedEnvTrans = envTransitionProbs[a_idx, SDEToNode]
        permutatedEnvTrans = permutatedEnvTrans[:, SDEToNode]

        # we divide by two so that way each transition diff is normalized to be between 0 and 1
        abs_difference = np.absolute(permutatedEnvTrans - modelTransitionProbs[a_idx,:,:]) / 2
        error = error + np.sum(np.sum(np.sum(abs_difference)))
        # import pdb; pdb.set_trace()

    error = error / (len(env.A_S) * num_of_env_states)
    return error

