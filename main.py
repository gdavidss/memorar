from Simulator import SRS_Simulator
import evalutation
from typing import List
from QLearning import QLearning
import numpy as np

NUM_CARDS = 4

def main():
    qLearning = QLearning(numCards=NUM_CARDS)
    simulator = SRS_Simulator(numCards=NUM_CARDS, model=qLearning)
    params: np.array = simulator.run(numEpisodes=1000)

    evalutation.evaluate(numCards=NUM_CARDS, numEpisodes=100000, params=params)

if __name__ == "__main__":
    main()