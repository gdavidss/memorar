from MDP import MDP
import random

class RandomPolicy(MDP):
    """"
    Class: RandomPolicy
    --------------------------
    This class implements an MDP problem where it chooses
    a random policy every time, regardless of the state.
    """
    def __init__(self, numCards: int):
        self.numCards = numCards

    def computePolicy(self, state) -> int:
        """Generates a random policy given number of cards"""
        return random.choice(range(self.numCards))