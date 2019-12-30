from abc import ABC, abstractmethod
import copy
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



# the set of observations
O = range(2)
# the set of actions
A = range(2)
#  the most likely state to transition to given T_ml[state][action]
T_ml = [[1, 2], [1, 3], [0, 0], [2, 1]]
#  the most likely observation given you're in that state
Omega_ml = [0, 0, 1, 1]
alpha = 0.99
epsilon = 1
E = POMDP(range(4), O, A, T_ml, Omega_ml, alpha, epsilon)

# print(E.getSDEDistribution([0, 1, 1, 0, 0]))
# exit()
# temp = E.getSDEConditionalProbability([1, 0, 1], 0, [1])
# print(temp)
# exit()

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
            total = sum(probabilities)
            for x in range(len(probabilities)):
                probabilities[x] = probabilities[x] / total

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
                os.system("pause")
                modelStates.append(m_1_new)
                modelStates.append(m_2_new)
                # TODO: append new SDE to the list of SDEs S
                modelStates.remove(m)
                change = True

                if change is True:
                    modified = True
                    break
                    break

print("Final states are: " + str(modelStates))
# TODO: print out the transition probabilities of the final graph