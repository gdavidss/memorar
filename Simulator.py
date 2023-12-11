import cardsDeck
from QLearning import QLearning
from UserModel import User, Card, Grade, Time
from typing import List, Tuple
import random
from collections import deque
from ExperienceReplay import ExperienceReplay
from tqdm import tqdm
from collections import defaultdict
import time

EPSILON = 0.99
BATCH_SIZE = 10

class SRS_Simulator():
    """
    Class: SRS_Simulator
    --------------------------
    This class implements a simulator for a Space Repetition System (SRS)
    """
    def __init__(self, model: QLearning, deck: cardsDeck, verbose: bool = False):
        self.deck = deck
        self.model = model
        self.numCards = self.deck.numCards
        self.epsilon = EPSILON
        self.epsilon_min = 0.1 # always explore at least 10 percent of the time
        self.epsilon_decay = 0.995 # 5 percent decay in each run
        self.batchSize = BATCH_SIZE
        self.experienceDB = ExperienceReplay()
        self.state: List[Card] = [(Grade.Easy, 0) for _ in range(self.numCards)]
        
    def _getAction(self, state: List[Card]) -> int:
        """
        This method utilizes the ε-greedy algorithm
        to choose an action at each time-step. 
        Returns the card index to be reviewed
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.numCards))
        
        return self.model.computePolicy(state)
    
    @staticmethod
    def computeReward(grade: Grade, t: Time) -> float:
        """
        This method computes the reward by applying the 
        reward function of the model.
        """
        return grade.value * t

    def run(self, numEpisodes: int) -> None:
        """
        This method runs the simulation by creating user models
        and exploring the state space. This does
        """
        # Create User

        user = User(numCards=self.numCards, deckStats = self.deck) # pass in card stability mean and std dev, map card to (mean, std dev)
        initial_user_stability = []
        initial_user_stability.append([card_tuple[0] for card_tuple in user.cards])
        #average_change_user_stability = initial_user_stability # change average change in user stability to dict and use default dict
        prevStabilityChange = 0
        user_flips_per_card = defaultdict(lambda: [0] * self.numCards)
        userIndex = 0
         
        # TODO: Adapt to generate multiple users
        for _ in tqdm(range(numEpisodes)): # every episode is looking at a single card
            if (user.hasAchievedMastery()):
                print("user achieved mastery!")
                user = User(numCards=self.numCards, deckStats = self.deck) # new user
                self.state: List[Card] = [(Grade.Easy, 0) for _ in range(self.numCards)] # reset states
                userIndex += 1
                initial_user_stability.append([card_tuple[0] for card_tuple in user.cards])
                #average_change_user_stability = initial_user_stability
                card_tuples = user.cards
                list_of_first_elements = [tup[0] for tup in card_tuples]
                initial_user_stability.append(elem for elem in list_of_first_elements)

            # Choose a card to review
            action = self._getAction(self.state)
            user_flips_per_card[userIndex][action] += 1

            stabilityChange = prevStabilityChange - float(user.cards[action][0])
            #average_change_user_stability[userIndex][action].append(stabilityChange)
            prevStabilityChange = stabilityChange

            # Review card
            grade = user.reviewCard(action)
            user.updateState(action)
            user.incrementTime()

            # Compute Reward
            _, currCardTime = self.state[action]
            reward = self.computeReward(grade, currCardTime)

            # Create next state
            nextState = self.state
            nextState = [(grade, t + 1) for grade, t in self.state]
            nextState[action] = (grade, 0)

            # Add to experience replay
            self.experienceDB.store((self.state, action, reward, nextState))

            # Update weights based on episode
            if (self.experienceDB.size() >= self.batchSize):
                batch = self.experienceDB.sampleBatch(self.batchSize)
                for state, action, reward, nextState in batch:
                    self.model.updateWeights(state, action, reward, nextState)
            else:
                self.model.updateWeights(self.state, action, reward, nextState)
            self.state = nextState
            
            # (update for epsilon greedy: epsilon decay after each run/episode:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        print("num users: ", userIndex+1)

        for userIndex, num_flips_per_card in user_flips_per_card.items():
            # average change in stability per iteration per card
            print(f"Num repeated for each card, User {userIndex}: {num_flips_per_card} with the following initial stability per card: {initial_user_stability[userIndex]}")
            #print(f"Average change in stability for each card per user: {average_change_user_stability[userIndex]}")

        return self.model.weights

