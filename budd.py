import numpy as np
from scipy.stats import dirichlet, entropy

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
        self.Node_Set.append(sPOMDPNode(Observation = "LRVDocked", Action_Dictionary = {"b": 7, "f": 4, "t": 1})) #state 0
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
    
    conservativeness_factor =  1#How much the entropy is scaled (the higher the #, the higher the penalty for an uncertain starting belief state. Set to 0; or 1 or greater)
    confidence_factor = 1000 #The number of confident experiments required until learning can end (i.e. what the minimum gamma sum is)

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

    Action_Gammas = np.ones((len(env.A_S),len(SDE_List),len(SDE_List)))

    first_Observations = [item[0] for item in SDE_List]

    OneStep_Gammas = np.ones((len(env.A_S),len(env.A_S),len(SDE_List),len(SDE_List),len(SDE_List))) #[gamma]aa'mm'm'' (formatted this way as we always know what action we took, but only have a belief over which model state we are in)

    # # convert gammas to transition probabilities
    Action_Probs = np.zeros((len(env.A_S),len(SDE_List),len(SDE_List)))
    for action in range(len(env.A_S)):
        for state in range(len(SDE_List)):
            Action_Probs[action, state, :] = dirichlet.mean(Action_Gammas[action, state, :])

    #Generate a belief mask for each SDE that indicates what the likelihood is of being in each state if an SDE were to be successful
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
        SDE_Chance[SDE_idx] = env.Alpha**(len(SDE_List[SDE_idx]))
        if num_Correspond > 1:
            SDE_Chance[(np.array(first_Observations) == SDE[0])] = (1-SDE_Chance[SDE_idx])/(num_Correspond-1)
        SDE_Chance[SDE_idx] = env.Alpha**(len(SDE_List[SDE_idx]))
        SDE_Chance = SDE_Chance/np.sum(SDE_Chance)
        SDE_Belief_Mask.append(SDE_Chance)

    #Initialize Belief State using first observation in the trajectory
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


    for Transition_Idx in range(len(Informed_Transition)//2):
        #Belief State
        Belief_Mask = np.zeros(len(SDE_List))
        Observation = Informed_Transition[Transition_Idx*2+2]
        Action = Informed_Transition[Transition_Idx*2+1]
        Previous_Belief_State = Belief_State.copy()
        
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
        nonzero_values = np.count_nonzero(Previous_Belief_State)
        if (nonzero_values == 1):
            entropy_scaling = 1
        else:
            entropy_scaling = 1 - entropy(Previous_Belief_State, base=nonzero_values)
        # Previous_Belief_State = Previous_Belief_State[:,np.newaxis]
        Belief_Count = np.dot(Previous_Belief_State[:,np.newaxis],Belief_State[np.newaxis, :]) * pow(entropy_scaling, conservativeness_factor)
        # if Transition_Idx < len(Informed_Transition)//4:
        if Transition_Idx < len(Informed_Transition):
            max_row = np.argmax(np.max(Belief_Count, axis=1))
            Belief_Count[np.arange(len(SDE_List)) != max_row, :] = 0


        Model_Action_Idx = env.A_S.index(Action)

        Action_Gammas[Model_Action_Idx,:] = Belief_Count + Action_Gammas[Model_Action_Idx,:]
        
        # convert gammas to transition probabilities
        Action_Probs = np.zeros((len(env.A_S),len(SDE_List),len(SDE_List)))
        for action in range(len(env.A_S)):
            for state in range(len(SDE_List)):
                Action_Probs[action, state, :] = dirichlet.mean(Action_Gammas[action, state, :])


        #Update one-step transition gammas
        if Transition_Idx > 0: #i.e. this is the second action or later in the current trajectory, so we can calculate a one-step transition
            prevObservation = Informed_Transition[Transition_Idx*2]
            prevAction = Informed_Transition[Transition_Idx*2-1]
            prevModel_Action_Idx = env.A_S.index(prevAction)

            #For each non-zero value in tMinus2BeliefState (i.e. m), calculate what the value of the beliefStates would be after taking two actions/transitions (a and a')
            #After each transition, mask states that don't correspond to the m' and m'' observations (which will be either a pre-processed SDE "observation" or an environment observation). Then normalize.
            Transition1_Belief_State = np.dot(tMinus2BeliefState, Action_Probs[prevModel_Action_Idx,:,:])
            Transition1_Belief_State = Transition1_Belief_State*prevBelief_Mask
            Transition1_Belief_State =  Transition1_Belief_State/np.sum(Transition1_Belief_State)

            Transition2_Belief_State = np.dot(Transition1_Belief_State, Action_Probs[Model_Action_Idx,:,:])
            Transition2_Belief_State = Transition2_Belief_State*Belief_Mask
            Transition2_Belief_State =  Transition2_Belief_State/np.sum(Transition2_Belief_State)
            
            # if prevAction == "west" and prevObservation == 1 and Action == "west" and Observation == 1:
            #     print(tMinus2BeliefState)
            #     print(prevAction)
            #     print(prevObservation)
            #     print(prevBelief_Mask)
            #     print(Transition1_Belief_State)
            #     print(Action)
            #     print(Observation)
            #     print(Transition2_Belief_State)
            #     exit()

            #Construct the m by m' by m'' matrix that will be used to update the OneStep_Gammas matrix.
            #This involves scaling the resulting transition2 belief state by the transition1 belief state and then normalizing the entire matrix
            
            #Create a replicated array of transition2 belief state - replicate M times and then that array M times
            # subMatrix = np.zeros((len(SDE_List),len(SDE_List),len(SDE_List)))
            subMatrix = np.array([[Transition2_Belief_State]*len(SDE_List)]*len(SDE_List))
            
            #Use each previous belief state to mask the transition array
            #Add 2 new axis to original belief state and one axis to the first transition belief state to get the proper multiplication
            tmp1 = tMinus2BeliefState.copy()
            tmp1 = tmp1[:,np.newaxis]
            tmp1 = tmp1[:,np.newaxis]
            subMatrix = subMatrix * tmp1
            tmp2 = Transition1_Belief_State.copy()
            tmp2 = tmp2[:,np.newaxis]
            subMatrix = subMatrix * tmp2
            subMatrix = subMatrix / (np.sum(subMatrix))

            # print(subMatrix)
            # print(":::::::::::::::::")
            OneStep_Gammas[prevModel_Action_Idx,Model_Action_Idx,:] = OneStep_Gammas[prevModel_Action_Idx,Model_Action_Idx,:] + subMatrix
            # print(OneStep_Gammas)
            # print(tMinus2BeliefState)
            # print(prevAction)
            # print(prevObservation)
            # print(Transition1_Belief_State)
            # print(Action)
            # print(Observation)
            # print(Transition2_Belief_State)
            # exit()

        tMinus2BeliefState = Previous_Belief_State.copy()
        prevBelief_Mask = Belief_Mask.copy()


        if Transition_Idx % 1000 == 0:
            print(Transition_Idx)
            # print("---")
            # print("Action_Gammas")
            # print(Action_Gammas)
            # print("***")
            # print("Action_Probs")
            # print(Action_Probs)
            # print("---")

        """print(Full_Transition)
        print(Informed_Transition)
        print(Previous_Belief_State)
        print(Action)
        print(Observation)
        """

        if((np.min(np.sum(Action_Gammas, axis=2)) / len(SDE_List)) >= confidence_factor):
            print("Finished early at action # " + str(Transition_Idx))
            break

    # print(Transition_Idx)
    print("---")
    # print("Action_Gammas")
    # print(Action_Gammas)
    # print("***")
    print("Action_Probs")
    print(Action_Probs)
    print("---")
            
    return (Belief_State, Action_Probs, Action_Gammas, OneStep_Gammas)




#Lines 6-19 of Algorithm 1. If splitting is successful, returns True and the new environment. Otherwise returns False and the previous environment.
def trySplitBySurprise(env, Action_Probs, Action_Gammas, surpriseThresh, OneStep_Gammas):
    didSplit = False
    newEnv = env

    oneStep_TransitionProbs = OneStep_Gammas / np.reshape(np.repeat(np.sum(OneStep_Gammas, axis = 4),len(env.SDE_Set),axis=3),OneStep_Gammas.shape)
    mSinglePrimeSum_aPrime = np.sum(OneStep_Gammas,axis = 4) #The total number of times the m' state is entered from state m under action a with respect to action a'
    mSinglePrimeSum = np.sum(mSinglePrimeSum_aPrime,axis = 0) #The total number of times the m' state is entered from state m under action a
    # mPrimeSum = np.sum(Action_Gammas, axis = 2) #The total number of times the action a is executed from state m
    mPrimeSum = np.sum(np.sum(mSinglePrimeSum, axis = 0), axis=0) #The total number of times the m' state is entered
    
    print("OneStep_Gammas")
    print(OneStep_Gammas)
    print("----------------------")
    print(mSinglePrimeSum_aPrime)
    print("&&&&&&&&&&&&&&")
    print(mSinglePrimeSum)
    print("^^^^^^^^^^^^^^^")
    print(mPrimeSum)
    print("00000000000000")
    # print(Action_Gammas)
    # print(np.sum(Action_Gammas))
    # print("?????????????")
    # print(OneStep_Gammas)
    # print(np.sum(OneStep_Gammas))

    entropyMA = np.zeros((len(env.A_S),len(env.SDE_Set))) #index as action, model number 
    gainMA = np.zeros((len(env.A_S),len(env.SDE_Set))) #index as action, model number
    sigmaMatrix = np.zeros((len(env.A_S),len(env.SDE_Set)))
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

                    if a == "east" and aPrime == "east" and m == ["nothing","east","nothing"]:
                        print("Double East:")
                        print(mPrime)
                        print(w_ma)
                        print(oneStepTransitionProb)
                        print(oneStep_TransitionEntropy)
                        print((w_ma * oneStep_TransitionEntropy))

                    if a == "west" and aPrime == "east" and m == ["nothing","east","nothing"]:
                        print("Single West:")
                        print(mPrime)
                        print(w_ma)
                        print(oneStepTransitionProb)
                        print(oneStep_TransitionEntropy)
                        print((w_ma * oneStep_TransitionEntropy))

            gainMA[aPrime_idx,mPrime_idx] = entropyMA[aPrime_idx,mPrime_idx] - sigma
            sigmaMatrix[aPrime_idx,mPrime_idx] = sigma
            print("sdfsdfsdf")
            print(w_maSum)

    print("_________________")
    print(sigmaMatrix)
    print("_________________")

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

    m1_prime = []
    m2_prime = []
    a_optimal = ""
    m_optimal = ""
    maxEntropy = 0
    for m_idx, m in enumerate(env.SDE_Set):
        for a_idx, a in enumerate(env.A_S):
            transitionSetProbs = Action_Probs[a_idx,m_idx,:]
            transitionSetEntropy = entropyMA[a_idx, m_idx]
            gammaSum = np.sum(Action_Gammas[a_idx, m_idx, :])

            # if transitionSetEntropy > surpriseThresh and (gammaSum > len(env.SDE_Set)*2):  #Check to see if entropy is high enough and that we actually updated these values by more than a small decimal
                # didSplit = True
                #Find m1_prime and m2_prime such that they match up to a first difference in observation



                # #Choose the SDE that has the highest entropy, as this indicates that more information must be learned for this transition.
                # SDE_List = env.get_SDE()

                #Only look at the pair with the maximum probabilities to determine if entropy is sufficient to split.
                # orderedVals = transitionSetProbs.copy()
                # orderedVals.sort()
                # prob1 = orderedVals[-1] #largest probability
                # prob2 = orderedVals[-2] #second largest probability
                # sde1_idx = np.where(transitionSetProbs == prob1)[0][0]
                # sde2_idx = np.where(transitionSetProbs == prob2)[0][0]
                # sde1 = SDE_List[sde1_idx]
                # sde2 = SDE_List[sde2_idx]

                # normalized_Probs = np.array([prob1, prob2]) / np.sum(np.array([prob1, prob2]))
                # relativeEntropy = np.sum(np.multiply(normalized_Probs,(np.log2(normalized_Probs)))) * -1
                # if relativeEntropy > maxEntropy:
                #     m1_prime = sde1
                #     m2_prime = sde2
                #     a_optimal = a
                #     m_optimal = m
                #     maxEntropy = relativeEntropy

    maxGainIndex = np.unravel_index(np.argmax(gainMA),gainMA.shape)
    if gainMA[maxGainIndex] > surpriseThresh:  #Check to see if gain is high enough
        transitionSetProbs = Action_Probs[maxGainIndex[0],maxGainIndex[1],:]
        orderedVals = transitionSetProbs.copy()
        orderedVals.sort()
        prob1 = orderedVals[-1] #largest probability
        prob2 = orderedVals[-2] #second largest probability
        sde1_idx = np.where(transitionSetProbs == prob1)[0][0]
        sde2_idx = np.where(transitionSetProbs == prob2)[0][0]
        m1_prime = env.get_SDE()[sde1_idx]
        m2_prime = env.get_SDE()[sde2_idx]
        m_optimal = env.get_SDE()[maxGainIndex[1]]
        a_optimal = env.A_S[maxGainIndex[0]]
    else:
        return (False,env) #did not find an SDE to split that had high enough entropy

    print("==============")
    print(m_optimal)
    print(a_optimal)
    print(maxGainIndex)
    print(m1_prime)
    print(m2_prime)
    print(transitionSetProbs)
    print("==============")

    if not m1_prime or not m2_prime:
        return (False,env) #did not find an SDE to split that had high enough entropy
   
    m1_new = [m_optimal[0]]
    m1_new.append(a_optimal)
    m1_new = m1_new + m1_prime
    m2_new = [m_optimal[0]]
    m2_new.append(a_optimal)
    m2_new = m2_new + m2_prime

    SDE_Set_new = env.get_SDE()

    if m1_new in env.SDE_Set and m2_new in env.SDE_Set:
        return (False, env) #trying to add two "new" SDEs that are already in the SDE list = failed to split

    didSplit = True
    outcomesToAdd = 0
    if not m1_new in env.SDE_Set:
        SDE_Set_new.append(m1_new)
        outcomesToAdd = outcomesToAdd + 1
    if not m2_new in env.SDE_Set:
        SDE_Set_new.append(m2_new)
        outcomesToAdd = outcomesToAdd + 1
    if outcomesToAdd > 1:
        SDE_Set_new.remove(m_optimal)

    print(m1_new)
    print(m2_new)
    
    newEnv = genericModel(env.O_S, env.A_S, env.State_Size, SDE_Set_new, env.Alpha, env.Epsilon, env.Node_Set)
    return (didSplit, newEnv)

#TODO: Need to update this once Dirichlet distributions are determined
def getModelEntropy(env, transitionProbs):
    maximum = 0
    for m_idx, m in enumerate(env.SDE_Set):
        for a_idx, a in enumerate(env.A_S):
            transitionSetProbs = transitionProbs[a_idx,m_idx,:]
            transitionSetEntropy = entropy(transitionSetProbs, base=len(env.SDE_Set))
            #TODO: change this ratio to be the correct ratio from equation 4
            ratio = 1/(len(env.SDE_Set) * len(env.A_S))
            transitionSetProbs = transitionProbs[a_idx,m_idx,:]
            transitionSetEntropy = np.sum(np.multiply(transitionSetProbs,(np.log(transitionSetProbs) / np.log(len(env.SDE_Set))))) * -1
            if transitionSetEntropy > maximum:
                maximum = transitionSetEntropy
    print(maximum)
    print(transitionProbs)
    return maximum


#Algorithm 3: Approximate sPOMPDP Learning.
def approximateSPOMDPLearning(env, entropyThresh, numSDEsPerExperiment, explore, surpriseThresh):
    #Initialize model

    while True:
        print(env.SDE_Set)
        (beliefState, probsTrans, actionGammas, OneStep_Gammas) = activeExperimentation(env, numSDEsPerExperiment, explore)
        print("||||||||||||||||||||")
        # print(OneStep_Gammas)
        print("||||||||||||||||||||")

        if getModelEntropy(env, probsTrans) < entropyThresh:#Done learning
            break

        (splitResult, env) = trySplitBySurprise(env, probsTrans, actionGammas, surpriseThresh, OneStep_Gammas)
        if not splitResult:
            print("Stopped because not able to split")
            break
        # input("Done with the current iteration. Press any key to begin the next iteration.")

    print(env.SDE_Set)


#The code for alogithm two is run below.  It is getting close to completion.  Just need to finish up the last steps.
if __name__ == "__main__":
    # env = Example1()
    # SDE_Num = 50000
    # explore = 0.05
    # (beliefState, probsTrans, actionGammas, OneStep_Gammas) = activeExperimentation(env, SDE_Num, explore)
    # print(probsTrans)

    env = Example4()

    entropyThresh = 0.35 #0.2 Better to keep smaller as this is a weighted average that can be reduced by transitions that are learned very well.
    surpriseThresh = 0 #0.4 for entropy splitting; 0 for one-step extension gain splitting
    numSDEsPerExperiment = 100000
    explore = 0.05
    approximateSPOMDPLearning(env, entropyThresh, numSDEsPerExperiment, explore, surpriseThresh)
