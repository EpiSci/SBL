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
        self.Alpha = 0.9
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
    SDE_Num = 2
    explore = 0.1
    Full_Transition = [Current_Observation]

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

    #Learn Transitions
    SDE_List = env.get_SDE()

    #Generate initial transition matrixes and intial belief state.
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

    #TODO: Update the Dirichlet Distribution