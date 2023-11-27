import random
from typing import List, Tuple

class SRS_Simulator:
    def __init__(self, num_cards: int, initial_retrieviability: float, dt: float, verbose: bool = False):
        self.verbose = verbose
        self.num_cards = num_cards


        last_review = [0.0 for _ in range(self.num_cards)]
        retrievability = [initial_retrieviability for _ in range(self.num_cards)]
        self.state = [retrievability, last_review]
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

    def review_card(self, card: int, success: bool) -> None:
        if success:
            # If the review of a card was successful, set its retrievability to 1
            self.state[0][card] = 1.0
        else:
            # INCOMPLETE
            # Ideally, we should be just decrease the stability of that card, 
            # so its retrievability decays faster
            pass

        # Decrease retrievability for unreviewed cards due to passage of time
        # we should be decreasing the retrievability based on 
        #### 1) time since last review
        #### 2) stability of the card

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