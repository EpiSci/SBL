from abc import ABC, abstractmethod


class POMDP(ABC):
    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text.  Instead of T(s, a, s') being  probability number for each
    state/action/state triplet, we instead have T(s, a) return a list of (p, s')
    pairs.  We also keep track of the possible states, terminal states, and
    actions for each state. [page 615]"""

    def __init__(self, init, S, A):
        self.A = A
        self.S = S
        # TODO: Probaly need to change
        self.init = init

    @abstractmethod
    def T(self, state, action):
        """Transition model.  From a state and an action, return a list
        of (result-state, probability) pairs."""
        pass

    @abstractmethod
    def Omega(self, state):
        """Observation model. For a given state, return a list of (observed state, probability)"""
        pass

    def actions(self, state):
        """Set of actions that can be performed in this state.  By default, a
        fixed list of actions. Override this
        method if you need to specialize by state."""
        return self.A