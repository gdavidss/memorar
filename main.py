from Simulator import SRS_Simulator
import Evaluator
from typing import List
from QLearning import QLearning
import numpy as np

NUM_CARDS = 4

def main():
    qLearning = QLearning(numStates=NUM_CARDS)
    simulator = SRS_Simulator(numCards=NUM_CARDS, model=qLearning)
    weights: np.ndarray = simulator.run(numEpisodes=100)

    print("Learned weights", weights)

    Evaluator.evaluate(numCards=NUM_CARDS, numEpisodes=1000, weights=weights)

if __name__ == "__main__":
    main()