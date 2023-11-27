from simulator import SRS_Simulator

user_one = SRS_Simulator(num_cards=3, initial_retrieviability=0.5, dt=0.1, verbose=True)

user_one.review_card(0, True)
user_one.review_card(0, True)
user_one.review_card(0, True)

num_cards = 3
