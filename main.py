from Simulator import SRS_Simulator
import Evaluator
from typing import List
from QLearning import QLearning
import numpy as np
from cardsDeck import CardsDeck

NUM_CARDS = 40

def main():
    cards = [5, 10, 15, 20, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    for c in cards:
        NUM_CARDS = c
        qLearning = QLearning(numStates=NUM_CARDS)
        simulator = SRS_Simulator(numCards=NUM_CARDS, model=qLearning)
        weights: np.ndarray = simulator.run(numEpisodes=100000)
        Evaluator.evaluate(numCards=NUM_CARDS, numEpisodes=100000, 
        weights=weights, numEpisodesTrain=100000)

if __name__ == "__main__":
    main()
