from QLearning import QLearning
from UserModel import User, Card, Grade, Time
from typing import List, Tuple
import random
from collections import deque
from ExperienceReplay import ExperieceReplay

EPSILON = 0.99
BATCH_SIZE = 10

class SRS_Simulator():
    """
    Class: SRS_Simulator
    --------------------------
    This class implements a simulator for a Space Repetition System (SRS)
    """
    def __init__(self, numCards: int, model: QLearning, verbose: bool = False):
        self.model = model
        self.numCards = numCards
        self.epsilon = EPSILON
        self.batchSize = BATCH_SIZE
        self.experienceDB = ExperieceReplay()
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
        user = User(self.numCards)
         
        # TODO: Adapt to generate multiple users
        for _ in range(numEpisodes):
            if (user.hasAchievedMastery()):
                user = User(self.numCards)
                self.state: List[Card] = [(Grade.Easy, 0) for _ in range(self.numCards)]
            # Choose a card to review
            action = self._getAction(self.state)

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
        
        return self.model.weights