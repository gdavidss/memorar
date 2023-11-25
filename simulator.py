import random

class SRS_Simulator:
    def __init__(self, cards, initial_state, dt):
        # cards are dictionary of card numbers and their dependencies
        self.cards = cards
        self.num_cards = len(cards)
        self.state = initial_state # initial retrievability of cards
        self.dt = dt  # time step
        self.t = 0   # global time
    
    def select_card(self):
        # Implement card selection policy
        # This will be given by our POMDP policy
        pass

    def update_state(self, card, success):
        if success:
            # If the review of a card was successful, set its retrievability to 1
            self.state[card] = 1.0
        else:
            # INCOMPLETE
            # Ideally, we should be just decrease the stability of that card, 
            # so its retrievability decays faster
            pass

        # Decrease retrievability for unreviewed cards due to passage of time
        # we should be decreasing the retrievability based on 
        #### 1) time since last review
        #### 2) stability of the card

    def review_success_probability(self, card):
        # if our card is dependent on some other, 
        # this should compute a probability based on the retrievability of our dependencies
        # otherwise, we should define some arbirary probability of getting the card right
        pass

    def step(self):
        # Choose a card to review
        card = self.select_card()
        
        # Calculate whether the review is successful
        prob = self.review_success_probability(card)
        success = (random.random() < prob)

        # Update state
        self.update_state(card, success)
        
        # Increment time
        self.t += self.dt

        return self.state