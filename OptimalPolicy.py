from MDP import MDP

class OptimalPolicy(MDP):
    """"
    Class: OptimalPolicy
    --------------------------
    This class implements an MDP problem that "cheats" by
    getting the rewards list for each state and maximazes it.
    """
    def __init__(self, numCards: int):
        self.numCards = numCards

    def computePolicy(self, state, rewardList) -> int:
        """Generates a the optimal policy given number of cards"""
        return rewardList.index(max(rewardList))