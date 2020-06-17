import numpy as np
from scipy.stats import dirichlet, entropy
import networkx as nx
import xlwt
import git  # Note: The name of the python library is GitPython
from test import writeNumpyMatrixToCSV
import pomdp
import csv

#Algorithm 2: Active Experimentation. Returns the belief state and transition probabilities.
#budd: True - transition updates will not perform "column updates" and only update the transition associated with the most likely belief state
#conservativeness_factor: How much the entropy is scaled (the higher the #, the higher the penalty for an uncertain starting belief state. Set to 1 or greater, or 0 to disable belief-state entropy penalty)
#confidence_factor: The number of confident experiments required until learning can end (i.e. what the minimum gamma sum is). Set to 1 or greater
#percentTimeofBudd: A number between 0 and 1. e.g. 0.9 means budd will work on the first 90% of the trajectory
#c: the csv writer object (if using writeToFile)
# Assuming AE environment with M states, the most likely transition should be around min{1 + (1-M)/(confidence_factor*M), alpha}
def activeExperimentation(env, numActions, explore, have_control, writeToFile, c, earlyTermination, budd, conservativeness_factor, confidence_factor, percentTimeofBudd, filename):
    Current_Observation = env.reset()

    SDE_List = env.get_SDE()

    #Generate Full Transitions
    Full_Transition = [Current_Observation]

    # make numActions equal to one if we have control that way we don't generate a large trajectory unnecessarily
    if have_control is False:
        #Execute numActions amount of SDEs. This is overkill, as not all of the SDEs will be used, but it doesn't add too much overhead
        for num in range(0,numActions):
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
            if num < numActions: #Insert random actions between each SDE
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
        # print(SDE_Chance)
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
    # while Transition_Idx < len(Informed_Transition)//2:
    while Transition_Idx < numActions: 
        # print(len(Informed_Transition))
        # print(Transition_Idx)
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
        if Transition_Idx < (numActions)*percentTimeofBudd and budd:
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
                modelNum = len(SDE_List) - len(env.O_S)
                c.writerow([])
                c.writerow(["Model Num " + str(modelNum)])

                c.writerow(["Model States: "])
                c.writerow(env.SDE_Set)
                
            if Transition_Idx % 5000 == 0 or Transition_Idx == numActions - 1:
                iterError = pomdp.calculateError(env, Action_Probs, 10000)
                c.writerow(["Iteration: ", Transition_Idx])
                c.writerow(["Error:", iterError])
                c.writerow(["Transition Probabilities"])
                writeNumpyMatrixToCSV(c, Action_Probs)

        #<<New Work: Implement a confidence factor that allows for early termination of the algorithm if each transition has been performed a reasonable # of times>>
        if((np.min(np.sum(Action_Gammas, axis=2)) / len(SDE_List)) >= confidence_factor) and earlyTermination:
            print("Finished early after " + str(Transition_Idx+1) + " actions")
            if writeToFile:
                iterError = pomdp.calculateError(env, Action_Probs, 10000)
                c.writerow(["Iteration: ", Transition_Idx])
                c.writerow(["Error:", iterError])
                c.writerow(["Transition Probabilities"])
                writeNumpyMatrixToCSV(c, Action_Probs)
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


#Lines 6-19 of Algorithm 1. If splitting is successful, returns True and the new environment. Otherwise returns False and the previous environment.
def trySplitBySurprise(env, Action_Probs, Action_Gammas, surpriseThresh, OneStep_Gammas, useEntropy):
    didSplit = False
    newEnv = env
    
    print("----- Trying Split By Surprise -----")
    print("OneStep_Gammas")
    print(OneStep_Gammas)

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
                    if np.where(transitionSetProbs == prob1)[0].size > 1: # In this case, the most likely probability actually occurs twice (e.g. a 50-50 transition split)
                        sde2_idx = np.where(transitionSetProbs == prob1)[0][1]
                    else:
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

    print("First new state: " + str(m1_new))
    print("Second new state :" + str(m2_new))
    
    newEnv = pomdp.genericModel(env.O_S, env.A_S, env.State_Size, SDE_Set_new, env.Alpha, env.Epsilon, env.Node_Set)
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
    print("Maximum transition entropy: " + str(maximum))
    print("Transiton Probabilities")
    print(transitionProbs)
    return maximum


#Algorithm 3: Approximate sPOMPDP Learning.
def approximateSPOMDPLearning(env, gainThresh, numActions, explore, surpriseThresh, splitWithEntropy=True, entropyThresh = 0.55, writeToFile=False,budd=True, earlyTermination=False,conservativeness_factor=0, confidence_factor=100,have_control=False, filename=None, percentTimeofBudd=0.9):

    if writeToFile:
        c = csv.writer(open(filename, "w", newline=''))
        # Write git repo sha
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        c.writerow(["github Code Version (SHA):", sha])

        # Write the training parameters
        parameterNames = ["Environment Observations","Environment Actions","alpha","epsilon", "numActions","explore","gainSplitThresh","surpiseThresh","splitWithEntropy", "entropyThresh","earlyTermination","budd","conservativeness_factor","confidence_factor","have_control"]
        parameterVals = [env.O_S, env.A_S, env.Alpha, env.Epsilon, numActions, explore, gainThresh, surpriseThresh, splitWithEntropy, entropyThresh, earlyTermination, budd, conservativeness_factor, confidence_factor, have_control]
        c.writerow(parameterNames)
        c.writerow(parameterVals)
    
    while True:
        print("|||||||||||||||||||| Learning Environment ||||||||||||||||||||")
        print("SDEs:")
        print(env.SDE_Set)
        
        (beliefState, probsTrans, actionGammas, OneStep_Gammas) = activeExperimentation(env=env, numActions=numActions, explore=explore, writeToFile=writeToFile, c=c,earlyTermination=earlyTermination,budd=budd,percentTimeofBudd=percentTimeofBudd,conservativeness_factor=conservativeness_factor, confidence_factor=confidence_factor, have_control=have_control, filename=filename)

        # print(OneStep_Gammas)

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

    print("Final learned SDE set:")
    print(env.SDE_Set)

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
