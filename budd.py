import numpy as np
from scipy.stats import dirichlet, entropy
import networkx as nx
import xlwt
import git


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


#Writes a numpy matrix to an xls file. Returns the last row the matrix was written on. Currently supports only 3D numpy matrices.
def writeNumpyMatrixToFile(sheet, matrix, row=0,col=0):
    dimensions = matrix.shape
    rowCount = row
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            
            for k in range(dimensions[2]):
                sheet.write(rowCount,col+k, matrix[i][j][k])
            rowCount = rowCount + 1
        rowCount = rowCount + 1 #Provide an extra space between submatrices
    return rowCount




#Algorithm 2: Active Experimentation. Returns the belief state and transition probabilities.
#budd: True - transition updates will not perform "column updates" and only update the transition associated with the most likely belief state
#conservativeness_factor: How much the entropy is scaled (the higher the #, the higher the penalty for an uncertain starting belief state. Set to 1 or greater, or 0 to disable belief-state entropy penalty)
#confidence_factor: The number of confident experiments required until learning can end (i.e. what the minimum gamma sum is). Set to 1 or greater
# Assuming AE environment with M states, the most likely transition should be around min{1 + (1-M)/(confidence_factor*M), alpha}
def activeExperimentation(env, SDE_Num, explore, have_control, writeToFile, earlyTermination, budd, conservativeness_factor, confidence_factor, workbook, filename):
    Current_Observation = env.reset()

    SDE_List = env.get_SDE()

    #Generate Full Transitions
    Full_Transition = [Current_Observation]

    # make SDE_Num equal to one if we have control that way we don't generate a large trajectory unnecessarily
    if have_control is False:
        #Execute SDE_Num amount of SDE.
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
    else:  # since we have control, just start off with an SDE to localize
        Matching_SDE = env.get_SDE(Current_Observation)
        Chosen_SDE = np.array(Matching_SDE[np.random.randint(low = 0, high = len(Matching_SDE))])
        Chosen_SDE_Actions = Chosen_SDE[np.arange(start=1, stop = len(Chosen_SDE), step= 2, dtype=int)]
        # if SDE is just an observation (and hence has no actions), just perform a random action
        if not Chosen_SDE_Actions:
            Current_Observation, random_action = env.random_step()
            Full_Transition.append(random_action)
            Full_Transition.append(Current_Observation)
        else:  # finish the SDE
            for action in Chosen_SDE_Actions:
                Current_Observation = env.step(action)
                Full_Transition.append(action)
                Full_Transition.append(Current_Observation)

    #Get SDE Transitions
    SDE_List = env.get_SDE()

    #<<New Work: Preprocess the trajectory>>
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

    #Learn Transitions

    #Initiate Transition Matrixes

    Action_Gammas = np.ones((len(env.A_S),len(SDE_List),len(SDE_List)))

    first_Observations = [item[0] for item in SDE_List]

    OneStep_Gammas = np.ones((len(env.A_S),len(env.A_S),len(SDE_List),len(SDE_List),len(SDE_List))) #[gamma]aa'mm'm'' (formatted this way as we always know what action we took, but only have a belief over which model state we are in)

    # convert gammas to transition probabilities
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

        num_Correspond = first_Observations.count(o)
        if num_Correspond > 1:
            SDE_Chance[(np.array(first_Observations) == SDE[0])] = (1 - env.Alpha**2)/(num_Correspond - 1)
        SDE_Chance[SDE_idx] = env.Alpha**2
        print(SDE_Chance)
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
                Belief_Mask[(np.array(first_Observations) == Observation)] = 1 
    else: #i.e. the array is all zeros and thus has not been changed - must be an SDE observation
        Belief_Mask = SDE_Belief_Mask[Observation]
    Belief_State = Belief_State*Belief_Mask
    Belief_State = Belief_State/np.sum(Belief_State)


    Transition_Idx = 0
    # print(Informed_Transition)
    # print(Belief_State)
    print(len(Informed_Transition))
    while Transition_Idx < len(Informed_Transition)//2:
        #<<New Work: Controlling the agent while generating the trajectory. This allows the agent to prioritize performing transitions it has yet to confidently learn>>

        #Create more trajectory if we have control and we're running out
        if have_control is True and len(Informed_Transition)//2 - Transition_Idx <= 1:

            new_Full_Transition = []

            # First predict where we'll be after the last action in the trajectory is performed
            Belief_Mask = np.zeros(len(SDE_List))
            Observation = Informed_Transition[Transition_Idx*2+2]
            Action = Informed_Transition[Transition_Idx*2+1]
            
            Model_Action_Idx = env.A_S.index(Action)
            Future_Belief_State = np.dot(Belief_State, Action_Probs[Model_Action_Idx,:,:])
        
            if Observation in env.O_S:
                for o in env.O_S:
                    if Observation == o:
                        Belief_Mask[(np.array(first_Observations) == Observation)] = 1 #If this is what I think it is, I think we should be using some function of alpha and/or epsilon...
            else: #i.e. the array is all zeros and thus has not been changed - must be an SDE observation
                Belief_Mask = SDE_Belief_Mask[Observation]

            Future_Belief_State = Future_Belief_State*Belief_Mask
            Future_Belief_State = Future_Belief_State/np.sum(Future_Belief_State)

            # perform localization if unsure of where we are
            nonzero_values = np.count_nonzero(Future_Belief_State)
            if entropy(Future_Belief_State, base=nonzero_values) > 0.75:
                # print("localizing")
                Matching_SDE = env.get_SDE(Current_Observation)
                Chosen_SDE = np.array(Matching_SDE[np.random.randint(low = 0, high = len(Matching_SDE))])
                Chosen_SDE_Actions = Chosen_SDE[np.arange(start=1, stop = len(Chosen_SDE), step= 2, dtype=int)]  # this is problematic in future as matching SDEs could have diff actions
                for action in Chosen_SDE_Actions:
                    Current_Observation = env.step(action)
                    new_Full_Transition.append(action)
                    new_Full_Transition.append(Current_Observation)

            else: # try to perform experiments so that we learn what we don't know

                # perform experiment if we're in a place where we can
                performed_experiment = False
                current_state = np.argmax(Future_Belief_State)
                for action_idx in range(len(env.A_S)):

                    if np.sum(Action_Gammas[action_idx, current_state])  / len(SDE_List) < confidence_factor:
                        action = env.A_S[action_idx]
                        Current_Observation = env.step(action)
                        new_Full_Transition.append(action)
                        new_Full_Transition.append(Current_Observation)
                        performed_experiment = True
                        # print("experiment performed: took action " + str(action) + " from state " + str(current_state))

                        # now localize again
                        Matching_SDE = env.get_SDE(Current_Observation)
                        Chosen_SDE = np.array(Matching_SDE[np.random.randint(low = 0, high = len(Matching_SDE))])
                        Chosen_SDE_Actions = Chosen_SDE[np.arange(start=1, stop = len(Chosen_SDE), step= 2, dtype=int)]  # this is problematic in future as matching SDEs could have diff actions
                        for action in Chosen_SDE_Actions:
                            Current_Observation = env.step(action)
                            new_Full_Transition.append(action)
                            new_Full_Transition.append(Current_Observation)

                        break

                # if unsuccesful, try to go to a state of interest
                if performed_experiment is False:
                    confidences = np.sum(Action_Gammas, axis=2) / len(SDE_List)
                    # TODO: Consider optimizing the chosen state based upon proximity

                    states_of_interest = np.array(np.where(confidences < confidence_factor))[1,:]
                    state_of_interest = states_of_interest[0]

                    G = getGraph(env, Action_Probs)
                    shortest_path = nx.dijkstra_path(G, current_state, state_of_interest, weight='weight')
                    # print("shortest_path")
                    # print(shortest_path)
                    # print("shortest path length")
                    # print(nx.dijkstra_path_length(G, current_state, state_of_interest, weight='weight'))
                    action_idx = np.argmax(Action_Probs[:,current_state, shortest_path[1]], axis=0)
                    action = env.A_S[action_idx]
                    Current_Observation = env.step(action)
                    new_Full_Transition.append(action)
                    new_Full_Transition.append(Current_Observation)
                    # print("Performing action " + str(action_idx) + " from state " + str(current_state) + " to get to state " + str(shortest_path[1]))

            # print("Informed_Transition before:")
            # print(Informed_Transition)

            #Detect Successful Transitions
            Full_Transition.extend(new_Full_Transition)
            Informed_Transition.extend(new_Full_Transition)
            for Transition_Num in range(Transition_Idx, len(Full_Transition)):
                for SDE_Idx, SDE in enumerate(SDE_List):
                    if len(SDE) <= len(Full_Transition)-Transition_Num:
                        for EO_Idx, Expected_Observation in enumerate(SDE):
                            if(Expected_Observation == Full_Transition[Transition_Num + EO_Idx]):
                                if EO_Idx == len(SDE)-1:
                                    Informed_Transition[Transition_Num] = SDE_Idx
                            else:
                                break
            # print("Informed_Transition after:")
            # print(Informed_Transition)

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
                    Belief_Mask[(np.array(first_Observations) == Observation)] = 1 #This may need to be some function of alpha and/or epsilon...
        else: #i.e. the array is all zeros and thus has not been changed - must be an SDE observation
            Belief_Mask = SDE_Belief_Mask[Observation]

        Belief_State = Belief_State*Belief_Mask
        Belief_State = Belief_State/np.sum(Belief_State)

        #<<New Work: Entropy scaling of belief state. Used to discourage learning from transitions if the initial belief state is not certain.>>
        #Updated Transition
        nonzero_values = np.count_nonzero(Previous_Belief_State)
        if (nonzero_values == 1):
            entropy_scaling = 1
        else:
            entropy_scaling = 1 - entropy(Previous_Belief_State, base=nonzero_values)
        # Previous_Belief_State = Previous_Belief_State[:,np.newaxis]
        Belief_Count = np.dot(Previous_Belief_State[:,np.newaxis],Belief_State[np.newaxis, :]) * pow(entropy_scaling, conservativeness_factor)

        #<<New Work: For first half of trajectory, only update the trans, only update the transition gammas for the transition that corresponds to the most likely starting state. This was done to avoid "column updates".>>
        if Transition_Idx < len(Informed_Transition)//4 and budd == True:
        #if Transition_Idx < len(Informed_Transition):
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

            OneStep_Gammas[prevModel_Action_Idx,Model_Action_Idx,:] = OneStep_Gammas[prevModel_Action_Idx,Model_Action_Idx,:] + subMatrix

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

        if writeToFile:
            if Transition_Idx == 0:
                rowIndex = 2
                errorRowIndex = 1
                errorColIndex = (len(SDE_List) * 2) + 7
                modelNum = len(SDE_List) - len(env.O_S)
                sh = workbook.add_sheet("Model " + str(modelNum))
                print("Adding workbook sheet " + "Model " + str(modelNum))
                sh.write(0,0, "Model States: ")
                sh.write(0,errorColIndex, "Action #:")
                sh.write(0,errorColIndex+1, "Error:")
                for SDE_id, SDE in enumerate(SDE_List):
                    sh.write(0,SDE_id+1, SDE)
            
            if Transition_Idx % 5000 == 0:
                sh.write(rowIndex, 0, "Transition Probabilities at Iteration: " +str(Transition_Idx))
                rowIndex = rowIndex + 1
                newRow = writeNumpyMatrixToFile(sh,Action_Probs,row=rowIndex,col=0)
                rowIndex = newRow + 2
                sh.write(errorRowIndex,errorColIndex, Transition_Idx)
                sh.write(errorRowIndex,errorColIndex+1, calculateError(env, Action_Probs, 10000))
                errorRowIndex = errorRowIndex + 1

            if Transition_Idx + 1 == len(Informed_Transition)//2: #The last action in the trajectory
                colIndex = 4+len(SDE_List)
                sh.write(0,colIndex, "Final Transition Probabilities")
                sh.write(0,colIndex+1, "Number of Actions:")
                sh.write(0,colIndex+2,Transition_Idx)
                sh.write(errorRowIndex,errorColIndex, Transition_Idx)
                sh.write(errorRowIndex,errorColIndex+1, calculateError(env, Action_Probs, 10000))
                newRow = writeNumpyMatrixToFile(sh,Action_Probs,row=1,col=colIndex)
                workbook.save(filename)
                print("Done writing to file")


        #<<New Work: Implement a confidence factor that allows for early termination of the algorithm if each transition has been performed a reasonable # of times>>
        if((np.min(np.sum(Action_Gammas, axis=2)) / len(SDE_List)) >= confidence_factor) and earlyTermination:
            print("Finished early after " + str(Transition_Idx+1) + " actions")
            if writeToFile:
                colIndex = 4+len(SDE_List)
                sh.write(0,colIndex, "Final Transition Probabilities")
                sh.write(0,colIndex+1, "Number of Actions:")
                sh.write(0,colIndex+2,Transition_Idx)
                newRow = writeNumpyMatrixToFile(sh,Action_Probs,row=1,col=colIndex)
                workbook.save(filename)
                print("Done writing to file")
            break




        Transition_Idx = Transition_Idx + 1

    # print(Transition_Idx)
    # print("---")
    # print("Action_Gammas")
    # print(Action_Gammas)
    # print("***")
    # print("Action_Probs")
    # print(Action_Probs)
    # print("---")
            
    return (Belief_State, Action_Probs, Action_Gammas, OneStep_Gammas)

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



#Lines 6-19 of Algorithm 1. If splitting is successful, returns True and the new environment. Otherwise returns False and the previous environment.
def trySplitBySurprise(env, Action_Probs, Action_Gammas, surpriseThresh, OneStep_Gammas, useEntropy):
    didSplit = False
    newEnv = env
    
    print("OneStep_Gammas")
    print(OneStep_Gammas)
    print("----------------------")


    m1_prime = []
    m2_prime = []
    a_optimal = ""
    m_optimal = ""
    
    if useEntropy:
        maxEntropy = 0
        for m_idx, m in enumerate(env.SDE_Set):
            for a_idx, a in enumerate(env.A_S):
                transitionSetProbs = Action_Probs[a_idx,m_idx,:]
                transitionSetEntropy = entropy(transitionSetProbs, base=len(env.SDE_Set))
                #TODO: change this ratio to be the correct ratio from equation 4
                if transitionSetEntropy > maxEntropy and transitionSetEntropy > surpriseThresh:
                    maxEntropy = transitionSetEntropy
                    transitionSetProbs = Action_Probs[a_idx,m_idx,:]
                    orderedVals = transitionSetProbs.copy()
                    orderedVals.sort()
                    prob1 = orderedVals[-1] #largest probability
                    prob2 = orderedVals[-2] #second largest probability
                    sde1_idx = np.where(transitionSetProbs == prob1)[0][0]
                    sde2_idx = np.where(transitionSetProbs == prob2)[0][0]
                    m1_prime = env.get_SDE()[sde1_idx]
                    m2_prime = env.get_SDE()[sde2_idx]
                    a_optimal = a
                    m_optimal = m
        print("Max Entropy: ")
        print(maxEntropy)

    else:
        (gainMA, entropyMA) = calculateGain(env, Action_Probs, OneStep_Gammas)
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

#NOTE: This function getModelEntropy is no longer used as it can't be generalized to non-alpha-epsilon environments. The gain values are now used to determine splitting
def getModelEntropy(env, transitionProbs):
    #<<New Work: Collins uses a weighted average for model entropy. We use the maximum transition entropy as this would scale better for larger environments with more transitions. Note that this code is not being used (as model splitting is now determined by the maximum gain in the environment).>>
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
def approximateSPOMDPLearning(env, gainThresh, numSDEsPerExperiment, explore, surpriseThresh, splitWithEntropy=True, entropyThresh = 0.55, writeToFile=False,budd=True, earlyTermination=False,conservativeness_factor=0, confidence_factor=100,have_control=False, filename=None):
    book = None
    #Initialize model
    while True:
        print(env.SDE_Set)
        if book == None and writeToFile:
            book = xlwt.Workbook()
            sh = book.add_sheet("Training Parameters")
            parameterDict = {"Environment Observations":env.O_S,"Environment Actions":env.A_S,"alpha":env.Alpha,"epsilon":env.Epsilon, "numSDEsPerExperiment":numSDEsPerExperiment,"SurpriseThresh":surpriseThresh,"explore":explore,"gainThresh":gainThresh,"splitWithEntropy":splitWithEntropy, "entropyThresh":entropyThresh, "earlyTermination":earlyTermination,"budd":budd,"conservativeness_factor":conservativeness_factor,"confidence_factor":confidence_factor,"have_control":have_control}
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            sh.write(0,0, "github Code Version (SHA): ")
            sh.write(0,1, sha)
            rowIndex = 1
            #Write parameters to the excel file
            for parameterName in parameterDict:
                sh.write(rowIndex,0, parameterName)
                if isinstance(parameterDict[parameterName],list):
                    for (item_index,item) in enumerate(parameterDict[parameterName]):
                        sh.write(rowIndex,1+item_index, item)
                else:
                    sh.write(rowIndex,1, parameterDict[parameterName])
                rowIndex=rowIndex+1
            book.save(filename)

        (beliefState, probsTrans, actionGammas, OneStep_Gammas) = activeExperimentation(env, numSDEsPerExperiment, explore, writeToFile=writeToFile, workbook=book, earlyTermination=earlyTermination,budd=budd,conservativeness_factor=conservativeness_factor, confidence_factor=confidence_factor, have_control=have_control, filename=filename)
        print("||||||||||||||||||||")
        # print(OneStep_Gammas)
        print("||||||||||||||||||||")

        gainMA = []
        if splitWithEntropy:
            if getModelEntropy(env, probsTrans) < entropyThresh:#Done learning
                break
        else:
            (gainMA, entropyMA) = calculateGain(env, probsTrans, OneStep_Gammas)
            #<<New Work: Use the maximum gain within the model to determine if the model should be split or not. This better generalizes to non alpha-epsilon environments>>
            if  np.max(gainMA) < gainThresh:#Done learning
                break

        (splitResult, env) = trySplitBySurprise(env, probsTrans, actionGammas, surpriseThresh, OneStep_Gammas, splitWithEntropy)
        if not splitResult:
            print("Stopped because not able to split")
            break
        # input("Done with the current iteration. Press any key to begin the next iteration.")

    print(env.SDE_Set)




# Calculate the error of a model as defined in Equation 6.4 (pg 122) of Collins' Thesis
# env holds the SDEs for the model
def calculateError(env, modelTransitionProbs, T):
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

    # Generate starting belief states for environment and model using first observation
    Observation = Full_Transition[0]
    Observation_Idx = env.O_S.index(Observation)
    Belief_State_mod = Obs_Belief_Mask_mod[Observation_Idx].copy()
    Belief_State_env = Obs_Belief_Mask_env[Observation_Idx].copy()

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
        error_vector = np.dot(Obs_Probs_mod, Belief_State_mod) - np.dot(Obs_Probs_env, Belief_State_env)
        error = error + np.sqrt(error_vector.dot(error_vector))

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




def getGraph(env, transitionProbs):
    SDE_List = env.get_SDE()
    G = nx.DiGraph()
    edges = []

    max_probs = np.max(transitionProbs, axis=0)

    for start in range(len(SDE_List)):
        for des in range(len(SDE_List)):
            edges.append((start, des, 1 - max_probs[start,des]))
    G.add_weighted_edges_from(edges)
    return G

#Uses the Test 1 parameters outlined in the SBLTests.docx file with column updates (Collins' method)
def test1_v1():
    #env = Example2()
    env = Example2()
    gainThresh = 0.05 #Threshold of gain to determine if the model should stop learning
    surpriseThresh = 0 #0; used for one-step extension gain splitting
    entropyThresh = 0.55
    numSDEsPerExperiment = 50000 #Note: for larger environments (e.g. Example5), this should be larger (e.g. 200,000)
    explore = 0.05
    approximateSPOMDPLearning(env, gainThresh, numSDEsPerExperiment, explore, surpriseThresh,splitWithEntropy=True, entropyThresh=entropyThresh, writeToFile=True, earlyTermination=False, budd=False, filename="Testing Data/Test1_v1_env2June3.xls")

#Uses the Test 1 parameters outlined in the SBLTests.docx file without column updates (Our method)
def test1_v2():
    env = Example2()
    gainThresh = 0.05 #Threshold of gain to determine if the model should stop learning
    surpriseThresh = 0 #0; used for one-step extension gain splitting
    entropyThresh = 0.55
    numSDEsPerExperiment = 50000 #Note: for larger environments (e.g. Example5), this should be larger (e.g. 200,000)
    explore = 0.05
    approximateSPOMDPLearning(env, gainThresh, numSDEsPerExperiment, explore, surpriseThresh, splitWithEntropy=True, entropyThresh=entropyThresh,writeToFile=True, earlyTermination=False, budd=True, filename="Testing Data/Test1_v2_env2June3.xls")

#Uses the Test 2 parameters outlined in the SBLTests.docx file with random actions (no agent control)
def test2_v1():
    env = Example2()
    gainThresh = 0.01 #Threshold of gain to determine if the model should stop learning
    surpriseThresh = 0 #0; used for one-step extension gain splitting
    entropyThresh = 0.4
    numSDEsPerExperiment = 50000 #Note: for larger environments (e.g. Example5), this should be larger (e.g. 200,000)
    explore = 0.05
    approximateSPOMDPLearning(env, gainThresh, numSDEsPerExperiment, explore, surpriseThresh, splitWithEntropy=True, entropyThresh=entropyThresh, writeToFile=True, earlyTermination=True, budd=True, have_control = False, conservativeness_factor=0, confidence_factor=5, filename="Testing Data/Test2_v1May26.xls")

#Uses the Test 2 parameters outlined in the SBLTests.docx file with agent control
def test2_v2():
    env = Example2()
    gainThresh = 0.01 #Threshold of gain to determine if the model should stop learning
    surpriseThresh = 0 #0; used for one-step extension gain splitting
    entropyThresh = 0.4
    numSDEsPerExperiment = 50000 #Note: for larger environments (e.g. Example5), this should be larger (e.g. 200,000)
    explore = 0.05
    approximateSPOMDPLearning(env, gainThresh, numSDEsPerExperiment, explore, surpriseThresh, splitWithEntropy=True, entropyThresh=entropyThresh, writeToFile=True, earlyTermination=True, budd=True, have_control = True, conservativeness_factor=0, confidence_factor=50, filename="Testing Data/Test2_v2May28.xls")

if __name__ == "__main__":
    # env = Example1()

    # SDE_Num = 1
    # explore = 0.05
    # (beliefState, probTrans, actionGammas, OneStep_Gammas) = activeExperimentation(env, SDE_Num, explore, writeToFile=False, workbook=None, earlyTermination=True,budd=True,conservativeness_factor=0, confidence_factor=10, have_control=True, filename=None)
    
    # (beliefState, probTrans, actionGammas, OneStep_Gammas) = activeExperimentation(env, 10000, explore, have_control=False)
    # print(probTrans)
    
    # print("probTrans")
    # print(probTrans)
    # print("Actual transitions:")
    # print(env.get_true_transition_probs())
    # error = calculateError(env, probTrans, T=1000)
    # error = calculateError(env, env.get_true_transition_probs(), T=1000)
    # print(error)

    # env = Example2()

    # # entropyThresh = 0.35 #0.2 Better to keep smaller as this is a weighted average that can be reduced by transitions that are learned very well.
    # gainThresh = 0.05 #Threshold of gain to determine if the model should stop learning
    # surpriseThresh = 0 #0.4 for entropy splitting; 0 for one-step extension gain splitting
    # numSDEsPerExperiment = 50000 #Note: for larger environments (e.g. Example5), this should be larger (e.g. 200,000)
    # explore = 0.05
    # approximateSPOMDPLearning(env, gainThresh, numSDEsPerExperiment, explore, surpriseThresh, writeToFile=True, earlyTermination=False, budd=True)
    test1_v2()