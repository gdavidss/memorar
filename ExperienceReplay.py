from UserModel import Card
from typing import List, Tuple
import random
from collections import deque

MAX_MEMORY_SIZE = 1000
Episode = Tuple[List[Card], int, float, List[Card]]

class ExperienceReplay():
    """
    Class: Experience Replay
    --------------------------
    This class implements a Experience Replay.
    """
    def __init__(self):
        self.memory = deque(maxlen=MAX_MEMORY_SIZE)

    def store(self, episode: Episode) -> None:
        """
        Store an episode in the experience replay memory.
        """
        self.memory.append(episode)
    
    def sampleBatch(self, batchSize: int) -> List[Episode]:
        """
        Sample a batch of episodes from the experience replay memory.
        """
        return random.sample(self.memory, batchSize)
    
    def size(self) -> int:
        """
        Get the current size of the experience replay memory.
        """
        return len(self.memory)