from UserModel import Card
from typing import List, Tuple
from MDP import MDP

#TODO: IMPLEMENT THIS

class QLearning(MDP):
    """
    Class: QLearning
    --------------------------
    This class implements an MDP problem utilizing QLearning with
    Function Approximation.
    """
    def __init__(self, numCards: int, params: List[float] = None):
        self.params = params
        pass

    def computePolicy(self, state: List[Card]) -> int:
        return 1
    
    def updateWeights(self, state: List[Card], action: int, reward: float, nextState: List[Card]):
        return