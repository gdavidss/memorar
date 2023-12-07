# Memorar: A Reinforcement Learning-Based Space Repetition System

## Description

Memorar is an intelligent and personalized Space Repetition System (SRS) that uses a Q-learning algorithm with function approximation for optimized memory retention.

## Code Structure

This project comprises several Python classes organized in different files. Below are the main files and the classes they contain:

### UserModel.py

The UserModel file contains classes for modeling a user's memory and grading system:

- `Grade`(Enum): Categorizes a learner's recall strength into 'Again', 'Hard', 'Medium', and 'Easy' based on retrieval probability.

- `User`: This Class models a user's memory using the Forgetting Curve concept. It maintains parameters like card stability and time since the last review and provides methods for reviewing a card, updating the card's state, increasing review time, and checking for mastery.

### QLearning.py

The `QLearning` class implements an MDP problem using Q-Learning with Function Approximation. It provides methods for computing the Q-value, updating the weights using gradient descent, and computing policy.

A relevant component of this class is the feature vector, `actionOneHotVector`, which uses one-hot encoding to represent the state-action pairs for input into the function approximator.

### SRS_Simulator.py

The `SRS_Simulator` class implements the learning environment. It uses the ε-Greedy policy for exploration and provides functionality for computing rewards and managing transition states.

### ExperienceReplay.py

The `ExperienceReplay` class manages the storage of past experiences or episodes, which are used for training the model. It uses a queue data structure to store episodes with a certain maximum size and provides functionalities for storing and sampling episodes.

### MDP.py, OptimalPolicy.py, RandomPolicy.py

These files implement different policies for the MDP problem:

- `MDP`(from MDP.py): An abstract base class that defines the interface for an MDP problem.

- `OptimalPolicy` (from OptimalPolicy.py): Implements an MDP problem where the policy is computed based on provided rewards.

- `RandomPolicy` (from RandomPolicy.py): Implements an MDP problem, where a policy is chosen randomly, independent of state.

### Testing Scripts

Evaluate.py are provided is generating test data, running simulations, evaluating the RL model, and comparing the model's performance against random and optimal policies.