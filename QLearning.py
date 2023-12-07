from UserModel import Card, Grade
from typing import List, Tuple, Optional
from MDP import MDP
import numpy as np

#TODO: IMPLEMENT THIS - CURRENT IMPLEMENTATION IS BROKEN

State = List[Card]
DEFAULT_ALPHA = 0.001
DEFAULT_DISCOUNT = 0.9

class QLearning(MDP):
    """
    Class: QLearning
    --------------------------
    This class implements an MDP problem utilizing QLearning with
    Function Approximation.
    """
    def __init__(self, numStates: int, weights: Optional[np.ndarray] = None, alpha: float = DEFAULT_ALPHA, discount: float = DEFAULT_DISCOUNT):
        self.numStates = numStates
        self.alpha = alpha
        self.discount = discount
        if weights is None:
            self.weights = np.zeros(shape=numStates + 1, dtype=float)
        else:
            self.weights = weights

    def featureVector(self, state: State, action: int):
        #stateWeightsVector = np.array([float(i.value) if type(i) == Grade else i for s in state for i in s])
        actionOneHotVector = np.array([G.value * t if i == action else 0 for i, (G, t) in enumerate(state)])
        return np.concatenate([[1], actionOneHotVector])

    def computeQ(self, state: State, action: int):
        return np.dot(self.weights, self.featureVector(state, action))

    def computePolicy(self, state: State) -> int:
        #print([self.computeQ(state, action) for action in range(self.numStates)])
        return np.argmax([self.computeQ(state, action) for action in range(self.numStates)])

    def computeGradient(self, weights, state, action):
        return self.featureVector(state, action)

    def scaleGradient(self, grad: np.ndarray, l2_max: float) -> np.ndarray:
        return min(l2_max / np.linalg.norm(grad), 1) * grad
    
    def updateWeights(self, state: State, action: int, reward: float, nextState: State):
        maxQPrime = np.max([self.computeQ(nextState, action) for action in range(self.numStates)])
        gradient = self.computeGradient(self.weights, state, action)
        delta = (reward + self.discount * maxQPrime - self.computeQ(state, action)) * gradient
        scaled_delta = self.scaleGradient(delta, 1.0)
        self.weights += self.alpha * scaled_delta