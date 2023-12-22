import random
import numpy as np
from typing import List, Tuple
from enum import Enum

import cardsDeck
import matplotlib.pyplot as plt

Stability = float
Time = float

class Grade(Enum):
    Again = 1 # Harder than "hard" (can't recall at all)
    Hard = 2
    Medium = 5
    Easy = 10

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
    def __init__(self, numCards: int, deckStats: cardsDeck, noise: bool = True) -> None:
        #print("mean stats: ", deckStats.mean_values)
        #print("std dev stats: ", deckStats.std_dev_values)
        if deckStats.isUniform:
            # model stability as uniform distribution
            self.cards: List[Card] = [(float(random.random()) if noise else DEFAULT_STABILITY, 0) for _ in
                                      range(numCards)]  # stability of memory

        else:
            # model stability as normal distribution

            self.cards: List[Card] = [(float(random.gauss(mean, std_dev)) if noise else DEFAULT_STABILITY, 0) for _, mean, std_dev in zip(range(numCards), deckStats.mean_values, deckStats.std_dev_values)]

            # see how mew and sigma change the curve online for normal distrb.

            # normal dist and randomly sample from that
            # randomly sample from that distribution



        self.stabilityScalingFactor: float = STABILITY_SCALING_FACTOR
    
    def _computeRetrievalibity(self, card: Card) -> float:
        """
        Computes the transition function for the user, 
        to ultimately return a grade based on the probability.
        The retrievability comes from the forgetting curve equation, found here: https://en.wikipedia.org/wiki/Forgetting_curve#cite_note-14
        """
        S, t = card
        return np.exp(-t/S)
    
    def _computeGrade(self, R: float) -> Grade:
        """
        Compute the discretized grade value on a card based on
        the retrivevability probability.
        """
        if R > 3/4:
            return Grade.Easy
        if R > 2/4:
            return Grade.Medium
        if R > 1/4:
            return Grade.Hard
        return Grade.Again
            
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
        # the smallest stability across all cards must be larger than threshold
        # if so user is "done" and we can generate next user
        return min([S for S, _ in self.cards]) > STABILITY_MASTERY
