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

#This class combines all of the nodes into a model.  This model is the one from figure 2.
class sPOMDPModelExample1():
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
        self.Node_Set.append(sPOMDPNode(Observation = self.SDE_Set[0][0], Action_Dictionary = {self.A_S[0]: 1, self.A_S[1]: 2})) #state 0
        self.Node_Set.append(sPOMDPNode(Observation = self.SDE_Set[1][0], Action_Dictionary = {self.A_S[0]: 1, self.A_S[1]: 3})) #state 1
        self.Node_Set.append(sPOMDPNode(Observation = self.SDE_Set[2][0], Action_Dictionary = {self.A_S[0]: 0, self.A_S[1]: 0})) #state 2
        self.Node_Set.append(sPOMDPNode(Observation = self.SDE_Set[3][0], Action_Dictionary = {self.A_S[0]: 2, self.A_S[1]: 1})) #state 3

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


#The code for alogithm two is run below.  It is getting close to completion.  Just need to finish up the last steps.
if __name__ == "__main__":
    env = sPOMDPModelExample1()
    Current_Observation = env.reset()

    #Generate Full Transitions
    SDE_Num = 1000
    explore = 0.05
    Full_Transition = [Current_Observation]

    X_Count = np.ones((4,4))*0.0001
    Y_Count = np.ones((4,4))*0.0001
    iterations = 1000

    lr = 0.01

    Old_X = None
    Old_Y = None

    for _ in range(iterations):
        print(_)
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

        X_Row_Sum = X_Count.sum(axis=1)
        Action_X = X_Count / X_Row_Sum[:, np.newaxis]
        Y_Row_Sum = Y_Count.sum(axis=1)
        Action_Y = Y_Count / Y_Row_Sum[:, np.newaxis]


        first_Observations = [item[0] for item in SDE_List]
        if (Old_X is not None) and (Old_Y is not None):
            Action_X = lr*Action_X + (1-lr)*Old_X
            Action_Y = lr*Action_Y + (1-lr)*Old_Y

        Old_X = Action_X
        Old_Y = Action_Y

        SDE_Belief_Mask = []
        for SDE_idx, SDE in enumerate(SDE_List):
            SDE_Chance = np.zeros(4)
            if SDE[0] == "square":
                SDE_Chance[0:2] = 0.5
            else:
                SDE_Chance[2:4] = 0.5
            """for o in env.O_S:
                #Set equal probability for each action to occur
                if SDE[0] == o:
                    #Figure out how many SDEs correspond to the observation
                    num_Correspond = first_Observations.count(o)
                    #Set the corresponding SDEs to 1 divided by that value
                    SDE_Chance[(np.array(first_Observations) == SDE[0])] = 1/num_Correspond"""
            

            Trans = np.ones((4,4))*0.25
            for action_idx in range(len(SDE)//2):
                Action = SDE[action_idx*2+1]
                Observation = SDE[action_idx*2+2]
                if Action == "x":
                    Tmp_Transition =  Action_X
                if Action == "y":
                    Tmp_Transition =  Action_Y
                Trans = np.dot(Trans, Tmp_Transition)
                #Mask the transition matrix
                if Observation == "square":
                    Trans[:,2:4] = 0
                else:
                    Trans[:,0:2] = 0
                """Trans[:,(np.array(first_Observations) != Observation)] = 0"""
                SDE_Chance = SDE_Chance * np.sum(Trans, axis=1)
            SDE_Chance[SDE_idx] = 0.99**2
            SDE_Chance = SDE_Chance/np.sum(SDE_Chance)
            SDE_Belief_Mask.append(SDE_Chance)

        #Initiate Belief State
        Belief_State = np.ones(len(SDE_List))/len(SDE_List)
        Belief_Mask = np.zeros(len(SDE_List))
        Observation = Informed_Transition[0]
        #If the observation is just a "simple" observation, then set the belief mask to 1 if the SDE for that state starts with that observation
        """if Observation in env.O_S:
            for o in env.O_S:
                if Observation == o:
                    Belief_Mask[(np.array(first_Observations) == Observation)] = 1 #If this is what I think it is, I think we should be using some function of alpha and/or epsilon...
        else: #i.e. the array is all zeros and thus has not been changed - must be an SDE observation
            Belief_Mask = SDE_Belief_Mask[Observation]"""
        if (Observation == "square") or (Observation == "diamond"):
            if Observation == "square":
                Belief_Mask[0:2] = 1
            else:
                Belief_Mask[2:4] = 1
        else:
            Belief_Mask = SDE_Belief_Mask[Observation]
        Belief_State = Belief_State*Belief_Mask
        Belief_State = Belief_State/np.sum(Belief_State)
                
        X_Count = np.zeros((4,4))+0.0001
        Y_Count = np.zeros((4,4))+0.0001

        for Transition_Idx in range(len(Informed_Transition)//2):
            #Belief State
            Belief_Mask = np.zeros(len(SDE_List))
            Observation = Informed_Transition[Transition_Idx*2+2]
            Action = Informed_Transition[Transition_Idx*2+1]
            Previous_Belief_State = Belief_State.copy()
            Previous_Belief_State = Previous_Belief_State[:,np.newaxis]

            if Action == "x":
                Belief_State = np.dot(Belief_State, Action_X)
            if Action == "y":
                Belief_State = np.dot(Belief_State, Action_Y)

            if (Observation == "square") or (Observation == "diamond"):
                if Observation == "square":
                    Belief_Mask[0:2] = 1
                else:
                    Belief_Mask[2:4] = 1
            else:
                Belief_Mask = SDE_Belief_Mask[Observation]
            Belief_State = Belief_State*Belief_Mask
            """if Observation in env.O_S:
                for o in env.O_S:
                    if Observation == o:
                        Belief_Mask[(np.array(first_Observations) == Observation)] = 1 #If this is what I think it is, I think we should be using some function of alpha and/or epsilon...
            else: #i.e. the array is all zeros and thus has not been changed - must be an SDE observation
                Belief_Mask = SDE_Belief_Mask[Observation]"""


            if np.sum(Belief_State) == 0:
                print(Informed_Transition[Transition_Idx*2-2:Transition_Idx*2+2])
                print(Belief_State)
                print(Action_X)
                print(Action_Y)
            Belief_State = Belief_State/np.sum(Belief_State)

            #Updated Transition
            Belief_Count = np.dot(Previous_Belief_State,Belief_State[np.newaxis, :])
            if Action == "x":
                X_Count = X_Count + Belief_Count
            #    X_Row_Sum = X_Count.sum(axis=1)
            #    Action_X = X_Count / X_Row_Sum[:, np.newaxis]
            if Action == "y":
                Y_Count = Y_Count + Belief_Count
            #    Y_Row_Sum = Y_Count.sum(axis=1)
            #    Action_Y = Y_Count / Y_Row_Sum[:, np.newaxis]
            
    print(Action_X)
    print(Action_Y)

"""     #Generate initial transition matrixes and intial belief state.
    Initial_Observation = Full_Transition[-1]
    State_Observations = []
    Belief_State = []
    for SDE in SDE_List:
        State_Observations.append(SDE[0])
        if SDE[0] == Initial_Observation:
            Belief_State.append(1)
        else:
            Belief_State.append(0)
    Belief_State = Belief_State/np.sum(Belief_State)


    #TODO: Update Transition Matrix Generation to include sub matrix identification.
    Transition_Matrix_List = {}
    for action in env.A_S:
        Transition_Matrix = []
        for SDE in SDE_List:
            Transition = []
            for Observation in State_Observations:
                if SDE[1] == action:
                    Transition.append(Observation == SDE[2])
                else :
                    Transition.append(1)
            Transition = Transition/np.sum(Transition)
            Transition_Matrix.append(Transition)
        Transition_Matrix_List[action] = np.array(Transition_Matrix)

    Reverse_Transition = np.flip(Full_Transition)

    #Calculate Belief State
    for idx in range(0, len(Reverse_Transition)-2, 2):
        Transition_Matrix = Transition_Matrix_List[Reverse_Transition[idx+1]]
        Belief_State = np.dot(Belief_State, np.transpose(Transition_Matrix))
        Belief_State = np.array(Belief_State * np.char.equal(State_Observations, Reverse_Transition[idx+2]))
        if np.sum(Belief_State) == 0:
            Belief_State += 1
        Belief_State = Belief_State / np.sum(Belief_State)
        print("======================")
        print(Transition_Matrix)
        print(Belief_State)

    #TODO: Update the Dirichlet Distribution """
