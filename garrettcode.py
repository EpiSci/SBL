import numpy as np

#The name of this file should be changed later.  I just wanted to make it obvious which file had my attempt.


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

#This class extends the generic sPOMDP model. This model is the one from figure 2.
class Example1(sPOMDPModelExample):
    def __init__(self):
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
        self.SDE_Set = []
        self.SDE_Set.append([self.O_S[0], self.A_S[1], self.O_S[1], self.A_S[0], self.O_S[0]])
        self.SDE_Set.append([self.O_S[0], self.A_S[1], self.O_S[1], self.A_S[0], self.O_S[1]])
        self.SDE_Set.append([self.O_S[1], self.A_S[0], self.O_S[0]])
        self.SDE_Set.append([self.O_S[1], self.A_S[0], self.O_S[1]])

        #Generate States
        self.Node_Set = []
        self.Node_Set.append(sPOMDPNode(Observation = "square", Action_Dictionary = {"x": 1, "y": 2})) #state 0
        self.Node_Set.append(sPOMDPNode(Observation = "square", Action_Dictionary = {"x": 1, "y": 3})) #state 1
        self.Node_Set.append(sPOMDPNode(Observation = "diamond", Action_Dictionary = {"x": 0, "y": 0})) #state 2
        self.Node_Set.append(sPOMDPNode(Observation = "diamond", Action_Dictionary = {"x": 2, "y": 1})) #state 3

#This class extends the generic sPOMDP model. This model is the from the environment from Figure 2, but only with 2 known SDEs (square or diamond)
class Example2(sPOMDPModelExample):
    def __init__(self):
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
        self.SDE_Set = []
        self.SDE_Set.append([self.O_S[0]])
        self.SDE_Set.append([self.O_S[1]])

        #Generate States
        self.Node_Set = []
        self.Node_Set.append(sPOMDPNode(Observation = "square", Action_Dictionary = {"x": 1, "y": 2})) #state 0
        self.Node_Set.append(sPOMDPNode(Observation = "square", Action_Dictionary = {"x": 1, "y": 3})) #state 1
        self.Node_Set.append(sPOMDPNode(Observation = "diamond", Action_Dictionary = {"x": 0, "y": 0})) #state 2
        self.Node_Set.append(sPOMDPNode(Observation = "diamond", Action_Dictionary = {"x": 2, "y": 1})) #state 3

#This class extends the generic sPOMDP model. This model is the from the environment from Figure 3, but only with 3 known SDEs (rose, volcano, or nothing)
class Example3(sPOMDPModelExample):
    def __init__(self):
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
        self.SDE_Set = []
        self.SDE_Set.append([self.O_S[0]])
        self.SDE_Set.append([self.O_S[1]])
        self.SDE_Set.append([self.O_S[2]])

        #Generate States
        self.Node_Set = []
        self.Node_Set.append(sPOMDPNode(Observation = "rose", Action_Dictionary = {"f": 3, "b": 2, "t": 0})) #state 0
        self.Node_Set.append(sPOMDPNode(Observation = "volcano", Action_Dictionary = {"f": 2, "b": 3, "t": 1})) #state 1
        self.Node_Set.append(sPOMDPNode(Observation = "nothing", Action_Dictionary = {"f": 0, "b": 1, "t": 3})) #state 2
        self.Node_Set.append(sPOMDPNode(Observation = "nothing", Action_Dictionary = {"f": 1, "b": 0, "t": 2})) #state 3

#This class extends the generic sPOMDP model. This model is the from the environment from Figure 4, but only with 2 known SDEs (goal, nothing)
class Example4(sPOMDPModelExample):
    def __init__(self):
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
        self.SDE_Set = []
        self.SDE_Set.append([self.O_S[0]])
        self.SDE_Set.append([self.O_S[1]])

        #Generate States
        self.Node_Set = []
        self.Node_Set.append(sPOMDPNode(Observation = "goal", Action_Dictionary = {"east": 1, "west": 3})) #state goal
        self.Node_Set.append(sPOMDPNode(Observation = "nothing", Action_Dictionary = {"east": 2, "west": 0})) #state left
        self.Node_Set.append(sPOMDPNode(Observation = "nothing", Action_Dictionary = {"east": 3, "west": 1})) #state middle
        self.Node_Set.append(sPOMDPNode(Observation = "nothing", Action_Dictionary = {"east": 0, "west": 2})) #state right

#This class extends the generic sPOMDP model. This model is the from the environment from Figure 5, 
#but only with 5 known starting SDEs (nothing, LRVForward, MRVForward, LRVDocked, MRVDocked)
class Example5(sPOMDPModelExample):
    def __init__(self):
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
        self.SDE_Set = []
        self.SDE_Set.append([self.O_S[0]])
        self.SDE_Set.append([self.O_S[1]])
        self.SDE_Set.append([self.O_S[2]])
        self.SDE_Set.append([self.O_S[3]])        
        self.SDE_Set.append([self.O_S[4]])

        #Generate States
        self.Node_Set = []
        self.Node_Set.append(sPOMDPNode(Observation = "LRVDocked", Action_Dictionary = {"b": 1, "f": 4, "t": 1})) #state 0
        self.Node_Set.append(sPOMDPNode(Observation = "MRVForward", Action_Dictionary = {"b": 1, "f": 1, "t": 4})) #state 1
        self.Node_Set.append(sPOMDPNode(Observation = "MRVForward", Action_Dictionary = {"b": 3, "f": 1, "t": 5})) #state 2
        self.Node_Set.append(sPOMDPNode(Observation = "nothing", Action_Dictionary = {"b": 0, "f": 2, "t": 6})) #state 3
        self.Node_Set.append(sPOMDPNode(Observation = "nothing", Action_Dictionary = {"b": 7, "f": 5, "t": 1})) #state 4
        self.Node_Set.append(sPOMDPNode(Observation = "LRVForward", Action_Dictionary = {"b": 4, "f": 6, "t": 2})) #state 5
        self.Node_Set.append(sPOMDPNode(Observation = "LRVForward", Action_Dictionary = {"b": 6, "f": 6, "t": 3})) #state 6
        self.Node_Set.append(sPOMDPNode(Observation = "MRVDocked", Action_Dictionary = {"b": 7, "f": 4, "t": 1})) #state 7

#Used in Algorithm 3 code as a generic model.
class genericModel(sPOMDPModelExample):
    def __init__(self, observationSet,actionSet, stateSize, SDE_Set, alpha, epsilon, environmentNodes):
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


#Algorithm 2: Active Experimentation. Returns the belief state and transition probabilities.
def activeExperimentation(env, SDE_Num, explore):
    Current_Observation = env.reset()

    SDE_List = env.get_SDE()

    #Generate Full Transitions
    Full_Transition = [Current_Observation]

    Action_Count = np.ones((len(env.A_S),len(SDE_List),len(SDE_List)))*0.0001
    iterations = 10000

    lr = 0.01

    Old_Action = [None for item in env.A_S]
    previousTransitions = []
    Action_Probs = np.array([])
    stopCount = 0

    for _ in range(iterations):
        if _ % 100 == 0:
           print(_)
           # if np.any(Action_Probs):
           #     print(Action_Probs)s

        if _ > 0:
            Full_Transition = [Full_Transition[-1]]
        #Exectue SDE_Num amount of SDE.
        for num in range(0,SDE_Num):
            Matching_SDE = env.get_SDE(Current_Observation)
            Chosen_SDE = np.array(Matching_SDE[np.random.randint(low = 0, high = len(Matching_SDE))])
            Chosen_SDE_Actions = Chosen_SDE[np.arange(start=1, stop = len(Chosen_SDE), step= 2, dtype=int)]
            for action in Chosen_SDE_Actions:
                if np.random.random() < explore:
                    Current_Observation, random_action = env.random_step()
                    Full_Transition.append(random_action)
                    Full_Transition.append(Current_Observation)
                    break
                else:
                    Current_Observation = env.step(action)
                    Full_Transition.append(action)
                    Full_Transition.append(Current_Observation)
            if num+1 < SDE_Num:
                Current_Observation, random_action = env.random_step()
                Full_Transition.append(random_action)
                Full_Transition.append(Current_Observation)

        #Get SDE Transitions
        SDE_List = env.get_SDE()

        #Detect Successful Transitions
        Informed_Transition = Full_Transition.copy()
        for Transition_Idx in range(len(Full_Transition)):
            for SDE_Idx, SDE in enumerate(SDE_List):
                if len(SDE) <= len(Full_Transition)-Transition_Idx:
                    for EO_Idx, Expected_Observation in enumerate(SDE):
                        if(Expected_Observation == Full_Transition[Transition_Idx + EO_Idx]):
                            if EO_Idx == len(SDE)-1:
                                Informed_Transition[Transition_Idx] = SDE_Idx
                        else:
                            break

        #The Code Below is a little hacky as it needs to be updated to work with any observation or action set.
        #Learn Transitions

        #Initiate Transition Matrixes

        Action_Row_Sum = Action_Count.sum(axis=2)
        Action_Probs = np.divide(np.transpose(Action_Count,(0,2,1)), Action_Row_Sum[:,np.newaxis])
        Action_Probs = np.transpose(Action_Probs,(0,2,1))

        first_Observations = [item[0] for item in SDE_List]
        if all(item is not None for item in Old_Action):
            Action_Probs = lr*Action_Probs + (1-lr)*Old_Action

        Old_Action = Action_Probs

        SDE_Belief_Mask = []
        for SDE_idx, SDE in enumerate(SDE_List):
            SDE_Chance = np.zeros(len(SDE_List))
            for o in env.O_S:
                #Set equal probability for each action to occur
                if SDE[0] == o:
                    #Figure out how many SDEs correspond to the observation
                    num_Correspond = first_Observations.count(o)
                    #Set the corresponding SDEs to 1 divided by that value
                    SDE_Chance[(np.array(first_Observations) == SDE[0])] = 1/num_Correspond
            

            Trans = np.ones((len(SDE_List),len(SDE_List)))/len(SDE_List)
            for action_idx in range(len(SDE)//2):
                Action = SDE[action_idx*2+1]
                Observation = SDE[action_idx*2+2]
                Model_Action_Idx = env.A_S.index(Action)
                Tmp_Transition = Action_Probs[Model_Action_Idx,:,:]
                Trans = np.dot(Trans, Tmp_Transition)
                #Mask the transition matrix
                Trans[:,(np.array(first_Observations) != Observation)] = 0
                SDE_Chance = SDE_Chance * np.sum(Trans, axis=1)
            SDE_Chance[SDE_idx] = 0.99**2
            SDE_Chance = SDE_Chance/np.sum(SDE_Chance)
            SDE_Belief_Mask.append(SDE_Chance)

        #Initiate Belief State
        Belief_State = np.ones(len(SDE_List))/len(SDE_List)
        Belief_Mask = np.zeros(len(SDE_List))
        Observation = Informed_Transition[0]
        #If the observation is just a "simple" observation, then set the belief mask to 1 if the SDE for that state starts with that observation
        if Observation in env.O_S:
            for o in env.O_S:
                if Observation == o:
                    Belief_Mask[(np.array(first_Observations) == Observation)] = 1 #If this is what I think it is, I think we should be using some function of alpha and/or epsilon...
        else: #i.e. the array is all zeros and thus has not been changed - must be an SDE observation
            Belief_Mask = SDE_Belief_Mask[Observation]
        Belief_State = Belief_State*Belief_Mask
        Belief_State = Belief_State/np.sum(Belief_State)
                
        Action_Count = np.ones((len(env.A_S),len(SDE_List),len(SDE_List)))*0.0001

        for Transition_Idx in range(len(Informed_Transition)//2):
            #Belief State
            Belief_Mask = np.zeros(len(SDE_List))
            Observation = Informed_Transition[Transition_Idx*2+2]
            Action = Informed_Transition[Transition_Idx*2+1]
            Previous_Belief_State = Belief_State.copy()
            Previous_Belief_State = Previous_Belief_State[:,np.newaxis]

            
            Model_Action_Idx = env.A_S.index(Action)
            Belief_State = np.dot(Belief_State, Action_Probs[Model_Action_Idx,:,:])
            
            if Observation in env.O_S:
                for o in env.O_S:
                    if Observation == o:
                        Belief_Mask[(np.array(first_Observations) == Observation)] = 1 #If this is what I think it is, I think we should be using some function of alpha and/or epsilon...
            else: #i.e. the array is all zeros and thus has not been changed - must be an SDE observation
                Belief_Mask = SDE_Belief_Mask[Observation]

            Belief_State = Belief_State*Belief_Mask
            Belief_State = Belief_State/np.sum(Belief_State)

            #Updated Transition
            Belief_Count = np.dot(Previous_Belief_State,Belief_State[np.newaxis, :])
            
            Model_Action_Idx = env.A_S.index(Action)
            Action_Count[Model_Action_Idx,:] = Action_Count[Model_Action_Idx,:] + Belief_Count
            """print(Full_Transition)
            print(Informed_Transition)
            print(Action_Count)
            print(Previous_Belief_State)
            print(Action)
            print(Observation)
            1/0"""

        if np.size(previousTransitions) > 0:
            delta = np.max(np.abs(previousTransitions - Action_Probs))
            if delta < 0.0005:
                stopCount = stopCount + 1
            else:
                stopCount = 0
            if stopCount >= 10:
                break

        previousTransitions = Action_Probs
            
    return (Belief_State, Action_Probs)




#Lines 6-19 of Algorithm 1. If splitting is successful, returns True and the new environment. Otherwise returns False and the previous environment.
def trySplitBySurprise(env, Action_Probs, surpriseThresh):
    didSplit = False
    newEnv = env

    # for action_idx, action in enumerate(env.A_S):
    #    for state_idx, transitionSetProbs in enumerate(Action_Probs[action_idx,:,:]):
    for m_idx, m in enumerate(env.SDE_Set):
        for a_idx, a in enumerate(env.A_S):
            transitionSetProbs = Action_Probs[a_idx,m_idx,:]
            transitionSetEntropy = np.sum(np.multiply(transitionSetProbs,(np.log(transitionSetProbs) / np.log(len(env.SDE_Set))))) * -1
            """print("-------------")
            print(m)
            print(a)
            print(transitionSetProbs)
            print(transitionSetEntropy)
            print("++++++++++++++")"""
            if transitionSetEntropy > surpriseThresh: 
                didSplit = True
                #Find m1_prime and m2_prime such that they match up to a first difference in observation
                #Choose the SDE that has the highest entropy, as this indicates that more information must be learned for this transition.
                SDE_List = env.get_SDE()
                maxEntropy = 0
                m1_prime = []
                m2_prime = []
                for sde1_idx, sde1 in enumerate(SDE_List):
                    for sde2_idx, sde2 in enumerate(SDE_List):
                        if (sde1 != sde2) and (sde1[:-1] == sde2[:-1]):
                            #Calculate relative entropy. The higher the entropy, the better to use these trajectories as an SDE.
                            prob1 = transitionSetProbs[sde1_idx]
                            prob2 = transitionSetProbs[sde2_idx]
                            normalized_Probs = np.array([prob1, prob2]) / np.sum(np.array([prob1, prob2]))
                            relativeEntropy = np.sum(np.multiply(normalized_Probs,(np.log2(normalized_Probs)))) * -1
                            if relativeEntropy > maxEntropy:
                                maxEntropy = relativeEntropy
                                m1_prime = sde1
                                m2_prime = sde2

                if not m1_prime or not m2_prime:
                    return (False,env) #Not sure if this case would ever occur, but if it does, return False
               
                m1_new = [m[0]]
                m1_new.append(a)
                m1_new = m1_new + m1_prime
                m2_new = [m[0]]
                m2_new.append(a)
                m2_new = m2_new + m2_prime

                SDE_Set_new = env.get_SDE()
                SDE_Set_new.append(m1_new)
                SDE_Set_new.append(m2_new)
                SDE_Set_new.remove(m)
                #Note: did note do line 14 of Algorithm 1 as this would add the new SDE to the SDE list.
                #       However, the way we currently store model states is by their SDE (i.e. actions and observations)
                #       The Collins paper stores the SDEs as only the corresponding actions

                
                newEnv = genericModel(env.O_S, env.A_S, env.State_Size, SDE_Set_new, env.Alpha, env.Epsilon, env.Node_Set)
                return (didSplit, newEnv)
    return (didSplit, newEnv)

#TODO: Need to update this once Dirichlet distributions are determined
def getModelEntropy(env, transitionProbs):
    summation = 0
    for m_idx, m in enumerate(env.SDE_Set):
        for a_idx, a in enumerate(env.A_S):
            transitionSetProbs = transitionProbs[a_idx,m_idx,:]
            transitionSetEntropy = np.sum(np.multiply(transitionSetProbs,(np.log(transitionSetProbs) / np.log(len(env.SDE_Set))))) * -1
            #TODO: change this ratio to be the correct ratio from equation 4
            ratio = 1/(len(env.SDE_Set) * len(env.A_S))
            summation = transitionSetEntropy*ratio + summation
    print(summation)
    print(transitionProbs)
    return summation


#Algorithm 3: Approximate sPOMPDP Learning.
def approximateSPOMDPLearning(env, entropyThresh, numSDEsPerExperiment, explore, surpriseThresh):
    #Initialize model

    while True:
        print(env.SDE_Set)
        (beliefState, probsTrans) = activeExperimentation(env, numSDEsPerExperiment, explore)

        if getModelEntropy(env, probsTrans) < entropyThresh:#Done learning
            break

        (splitResult, env) = trySplitBySurprise(env, probsTrans, surpriseThresh)
        if not splitResult:
            print("Stopped because not able to split")
            break
        # input("Done with the current iteration. Press any key to begin the next iteration.")

    print(env.SDE_Set)


#The code for alogithm two is run below.  It is getting close to completion.  Just need to finish up the last steps.
if __name__ == "__main__":
    """env = Example2()
    SDE_Num = 10
    explore = 0.05
    (beliefState, probsTrans) = activeExperimentation(env, SDE_Num, explore)
    print(probsTrans)"""

    env = Example5()

    entropyThresh = 0.2 #Better to keep smaller as this is a weighted average that can be reduced by transitions that are learned very well.
    surpriseThresh = 0.4
    numSDEsPerExperiment = 1000
    explore = 0.05
    approximateSPOMDPLearning(env, entropyThresh, numSDEsPerExperiment, explore, surpriseThresh)
