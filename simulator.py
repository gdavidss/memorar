import random, math
import numpy as np
from typing import List, Tuple


DEFAULT_STABILITY = 1.0

RETRIEVABILITY_INDEX = 0
STABILITY_INDEX = 1
LAST_REVIEW_INDEX = 2
NUM_REVIEWS_INDEX = 3

class SRS_Simulator:
    def __init__(self, num_cards: int, initial_retrieviability: float, dt: float, verbose: bool = False):
        self.verbose = verbose
        self.num_cards = num_cards

        last_review = np.array([0.0 for _ in range(self.num_cards)])
        retrievability = np.array([initial_retrieviability for _ in range(self.num_cards)])
        stability = np.array([0.5 for _ in range(self.num_cards)])
        num_reviews = np.array([0 for _ in range(self.num_cards)])

        self.state = np.array([retrievability, stability, last_review, num_reviews])
        self.dt = dt  # time step
        self.t = 0   # global time

        if self.verbose:
            print("Initialized SRS Simulator with {} cards".format(self.num_cards)
                    + " and initial retrievability {}".format(initial_retrieviability)
                    + " and time step {}".format(self.dt))
            print("Initial state: {}".format(self.state))

    def select_card(self) -> int:
        # Implement card selection policy
        # This will be given by our MDP policy
        pass

    def tick(self) -> None:
        self.t += self.dt

        # Update retrievability of all cards


    def review_card(self, card: int, success: bool) -> None:
        self.tick()
    
        if self.verbose:
            print("====================================")
            print("Reviewing card {}".format(card))
            print("Retrievability: {}".format(self.state[RETRIEVABILITY_INDEX][card]))
            print("Stability: {}".format(self.state[STABILITY_INDEX][card]))
        
        self.state[NUM_REVIEWS_INDEX][card] += 1
        self.state[RETRIEVABILITY_INDEX][card] = 1.0

        if success:
            # Improve stability of the card
            num_reviews = self.state[NUM_REVIEWS_INDEX][card]
            dt_card = self.t - self.state[LAST_REVIEW_INDEX][card]
            old_stability = self.state[STABILITY_INDEX][card]

            # how fast the stability grows in proportion to # of reviews
            # QUESTION: is it better for this to be nonlinear / exponential?
            scaling_factor = 0.04 
            new_stability = old_stability + (scaling_factor * num_reviews)
        
            if self.verbose:
                print("Card {} successfully reviewed".format(card))
                print("Retrievability: {}".format(self.state[RETRIEVABILITY_INDEX][card]))
                print("Stability: {}".format(new_stability))
        else:
            # let's do nothing with stability for now, just decrease retrievability with time
            pass
        
        # Update last review time
        self.state[LAST_REVIEW_INDEX][card] = self.t

    def review_success_probability(self, card: int) -> float:
        # if our card is dependent on some other, 
        # this should compute a probability based on the retrievability of our dependencies
        # otherwise, we should define some arbitrary probability of getting the card right
        pass

    def step(self) -> List[List[float]]:
        # Choose a card to review
        card = self.select_card()
        
        # Calculate whether the review is successful
        prob = self.review_success_probability(card)
        success = (random.random() < prob)

        # Update state
        self.review_card(card, success)
        
        # Increment time
        self.t += self.dt

        return self.state