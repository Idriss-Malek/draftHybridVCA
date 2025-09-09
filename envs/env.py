from abc import ABC, abstractmethod

class Env(ABC):
    @abstractmethod
    def neighbor(self, *args):
        raise NotImplemented
    
    @abstractmethod
    def energy(self, *args):
        raise NotImplemented
    
    @abstractmethod
    def canonicalize(self, input):
        """
        Let a be an action, and assume that energy is invariant under this action.
        This function return the canonical representation of the orbit of the input.
        This allow us to work on the quotient space instead of using costly geoDL. 
        """
        return input

    @abstractmethod
    def behaviorAR(self, input):
        """
        What is the behavior of the environment during the autoregressive process. 
        In other words, what is the mask on possible actions during the next step.
        No restrictions means a full True tensor.
        """
        return None