import random
import numpy as np
from typing import List, Tuple
from enum import Enum

Stability = float
Time = float
class Grade(Enum):
    Hard = 0
    Medium = 1
    Easy = 3
Card = Tuple[Stability, Time] 
STABILITY_MASTERY = 2000

STABILITY_SCALING_FACTOR = 1.1
DEFAULT_STABILITY = 5
DELTA_T = 1

class User:
    """
    Class: User
    --------------------------
    This class implements the model of the user memory by utilizing
    the Forgetting Curve, where it keeps track of the stability of 
    memory (S) and time since last review for each card (t).
    """
    def __init__(self, numCards: int, noise: bool = True) -> None:
        self.cards: List[Card] = [(float(random.random()) if noise else DEFAULT_STABILITY, 0) for _ in range(numCards)] # stability of memory
        self.stabilityScalingFactor: float = STABILITY_SCALING_FACTOR
    
    def _computeRetrievalibity(self, card: Card) -> float:
        """
        Computes the transition function for the user, 
        to ultimately return a grade based on the probability
        """
        S, t = card
        return np.exp(-t/S)
    
    def _computeGrade(self, R: float) -> Grade:
        """
        Compute the discretized grade value based on 
        the retrivevability probability.
        """
        if R > 2/3:
            return Grade.Easy
        if R > 1/3:
            return Grade.Medium
        return Grade.Hard

    def reviewCard(self, cardIndex: int) -> Grade:
        """
        Reviews the card given a card index, updates the 
        stability and time since last review, and returns
        a grade based on the current retrievability of that
        card. 
        """
        card = self.cards[cardIndex]
        R = self._computeRetrievalibity(card)
        #print("Card:", cardIndex, " | S:", card[0], " | t:", card[1], " | R:", R)

        # Get Grade
        grade = self._computeGrade(R)

        return grade
    
    def updateState(self, cardIndex: int) -> List[Card]:
        # Update card stability and last reviewed time
        currentS, _ = self.cards[cardIndex]
        self.cards[cardIndex] = (currentS * self.stabilityScalingFactor, 0)
        return self.cards

    def incrementTime(self) -> None:
        """
        Increase time since last reviewed for all
        cards.
        """
        self.cards = [(S, t + DELTA_T) for S, t in self.cards]

    def hasAchievedMastery(self) -> bool:
        return min([S for S, _ in self.cards]) > STABILITY_MASTERY