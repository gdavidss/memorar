from typing import List
from QLearning import QLearning
from UserModel import User, Card, Grade
from Simulator import SRS_Simulator
from RandomPolicy import RandomPolicy
from OptimalPolicy import OptimalPolicy
from typing import List, Tuple
from MDP import MDP
import pickle
import random

TEST_DATA_FILENAME = "testingData.pkl"

def generateTestData(numEpisodes: int, numCards: int) -> List[Tuple[List[Card], List[float]]]:
    """
    This function simulates a user and generates test data from scratch. It does
    so by going through the number of specified episodes and computes a new state
    a reward list for each possible action in that state.
    """
    user = User(numCards=numCards, noise=False)

    testingData: List[Tuple(List[Card], List[float])] = []
    state: List[Card] = [(Grade.Easy, 0) for _ in range(numCards)]

    # TODO: Possibly change this to simulate multiple users
    for _ in range(numEpisodes):
        # Choose a card to review
        rewardList = [SRS_Simulator.computeReward(user.reviewCard(action), t) for action, (_, t) in enumerate(state)]

        action = random.choice(range(numCards))
        grade = user.reviewCard(action)
        user.updateState(action)
        user.incrementTime()

        testingData.append((state, rewardList))
        state = [(grade, t + 1) for grade, t in state]
        state[action] = (grade, 0)

    return testingData

def getTestData(numEpisodes: int, numCards: int, force: bool = False) -> List[Tuple[List[Card], List[float]]]:
    """
    This function gets the testing data to evaluate the model. This data
    can come from a file or be generated if force is true or a file doesn't
    exist. 
    """
    if force:
        testingData = generateTestData(numEpisodes=numEpisodes, numCards=numCards)
        with open(TEST_DATA_FILENAME, 'wb') as file:
            pickle.dump(testingData, file)
        return testingData

    try:
        with open(TEST_DATA_FILENAME, 'rb') as file:
            testingData = pickle.load(file)
    except (FileNotFoundError, EOFError) as e:
        testingData = generateTestData(numEpisodes=numEpisodes, numCards=numCards)
        with open(TEST_DATA_FILENAME, 'wb') as file:
            pickle.dump(testingData, file)

    return testingData

def evaluateModel(model: MDP, testingData: List[Tuple[List[Card], List[float]]]) -> float:
    """
    This function evaluates the model by going through each 
    state in the testing data and choosing a policy. The utility
    (or score) is the cumulative sum of rewards of all datapoints.
    """
    score = 0

    for state, rewardList in testingData:
        if type(model) == OptimalPolicy:
            policy = model.computePolicy(state=state, rewardList=rewardList)
        else:
            policy = model.computePolicy(state=state)
        score += rewardList[policy]
    
    return score

def evaluate(numCards: int, numEpisodes: int, params: List[float]):
    """
    Runs a number of simulations and compare the total
    sum of rewards (utility) of our model with a random policy.
    """
    testingData = getTestData(numEpisodes=numEpisodes, numCards=numCards, force=False)
    qLearningScore = evaluateModel(QLearning(numCards=numCards), testingData=testingData)
    randomScore = evaluateModel(RandomPolicy(numCards=numCards), testingData=testingData)
    optimalScore = evaluateModel(OptimalPolicy(numCards=numCards), testingData=testingData)

    # Print results
    print(f"**Results from evalutation:**")
    print(f"Model Utility: {qLearningScore}")
    print(f"Optimal Policy Utility: {optimalScore}")
    print(f"Random Policy Utility: {randomScore}")
    print(f"Score (model - random): {qLearningScore - randomScore}")

