from Simulator import SRS_Simulator
import Evaluator
from typing import List
from QLearning import QLearning
import numpy as np
from cardsDeck import CardsDeck

NUM_CARDS = 4

def main():

    qLearning = QLearning(numStates=NUM_CARDS)

    uniform_deck = CardsDeck(numCards = NUM_CARDS, isUniform = True)
    simulator = SRS_Simulator(model=qLearning, deck = uniform_deck)
    weights: np.ndarray = simulator.run(numEpisodes=10000)
    print("Learned weights", weights)
    print("When modeling stability as a uniform distribution:")
    Evaluator.evaluate(numCards=NUM_CARDS, numEpisodes=10000, weights=weights, deck = uniform_deck)

    gaussian_deck = CardsDeck(numCards=NUM_CARDS, isUniform=False)
    simulator = SRS_Simulator(model=qLearning, deck = gaussian_deck)
    weights: np.ndarray = simulator.run(numEpisodes=10000)
    print("Learned weights", weights)
    print("When modeling stability as a normal distribution:")
    Evaluator.evaluate(numCards=NUM_CARDS, numEpisodes=10000, weights=weights, deck = gaussian_deck)

    # graph time until mastery (that describes a deck of cards)
    # graph utility

if __name__ == "__main__":
    main()
