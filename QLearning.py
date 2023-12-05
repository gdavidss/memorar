from UserModel import Card
from typing import List, Tuple
from MDP import MDP
import numpy as np

#TODO: IMPLEMENT THIS

State = List[Card]
DEFAULT_ALPHA = 0.01
DEFAULT_DISCOUNT = 0.9

class QLearning(MDP):
    """
    Class: QLearning
    --------------------------
    This class implements an MDP problem utilizing QLearning with
    Function Approximation.
    """
    def __init__(self, numStates: int, alpha: float = DEFAULT_ALPHA, discount: float = DEFAULT_DISCOUNT, weights: np.array = None):
        self.numStates = numStates
        self.alpha = alpha
        self.discount = discount
        if not weights:
            self.weights = np.zeros(3 * numStates + 1)
        else:
            self.weights = weights

    def featureVector(self, state: State, action: int):
        stateWeightsVector = [i for s in state for i in s]
        actionOneHotVector = [1 if i == action else 0 for i in range(self.numStates)]
        return np.concatenate([1], stateWeightsVector, actionOneHotVector)

    def computeQ(self, state: State, action: int):
        return np.dot(self.weights, self.featureVector(state, action))

    def computePolicy(self, state: State) -> int:
        return np.argmax([self.computeQ(state, action) for action in range(self.numStates)])
    
    def updateWeights(self, state: State, action: int, reward: float, nextState: State):
        self.weights += self.alpha * (reward + self.discount * np.max([self.computeQ(nextState, action) for action in range(self.numStates)]) - self.computeQ(state, action))