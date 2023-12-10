from Simulator import SRS_Simulator
import Evaluator
from typing import List
from QLearning import QLearning
import numpy as np

NUM_CARDS = 4

def main():
    qLearning = QLearning(numStates=NUM_CARDS)
    simulator = SRS_Simulator(numCards=NUM_CARDS, model=qLearning, uniform = True)
    weights: np.ndarray = simulator.run(numEpisodes=10000)
    print("Learned weights", weights)
    print("When modeling stability as a uniform distribution:")
    Evaluator.evaluate(numCards=NUM_CARDS, numEpisodes=10000, weights=weights)

    simulator = SRS_Simulator(numCards=NUM_CARDS, model=qLearning, uniform= False)
    weights: np.ndarray = simulator.run(numEpisodes=10000)
    print("Learned weights", weights)
    print("When modeling stability as a normal distribution:")
    Evaluator.evaluate(numCards=NUM_CARDS, numEpisodes=10000, weights=weights)

if __name__ == "__main__":
    main()
