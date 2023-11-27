import random, math
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

        last_review = [0.0 for _ in range(self.num_cards)]
        retrievability = [initial_retrieviability for _ in range(self.num_cards)]
        stability = [1.0 for _ in range(self.num_cards)]
        num_reviews = [0 for _ in range(self.num_cards)]

        self.state = [retrievability, stability, last_review, num_reviews]
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

    def review_card(self, card: int, success: bool) -> None:
        self.state[NUM_REVIEWS_INDEX][card] += 1
        
        if success:
            # If the review of a card was successful, set its retrievability to 1
            self.state[RETRIEVABILITY_INDEX][card] = 1.0

            # Improve stability of the card
            num_reviews = self.state[NUM_REVIEWS_INDEX][card]
            dt_card = self.state[LAST_REVIEW_INDEX][card] - self.t
            growth_factor = 1 # how fast the stability grows in proportion to # of reviews
            new_stability = math.e ** (-dt_card / (math.e ** (growth_factor * num_reviews)))
            self.state[STABILITY_INDEX][card] = new_stability
        else:
            # INCOMPLETE
            # Ideally, we should be just decrease the stability of that card, 
            # so its retrievability decays faster

            # let's do nothing for now though, we can experiment decreasing it later
            pass
        
        # Update last review time
        self.state[LAST_REVIEW_INDEX][card] = self.t

    def review_success_probability(self, card: int) -> float:
        # if our card is dependent on some other, 
        # this should compute a probability based on the retrievability of our dependencies
        # otherwise, we should define some arbitrary probability of getting the card right
        pass

    def step(self) -> List[List[float]]:
        self.tick()
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