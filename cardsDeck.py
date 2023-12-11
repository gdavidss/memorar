import random
class CardsDeck:
    def __init__(self, numCards: int, isUniform: bool = False, mean_values = [], std_dev_values = []) -> None:

        mean_values = []
        std_dev_values = []

        for _ in range(numCards):
            mean_values.append(random.uniform(0.2, 0.6))
            std_dev_values.append(random.uniform(0.1, 0.4))
        self.mean_values = mean_values
        self.std_dev_values = std_dev_values
        self.numCards = numCards
        self.isUniform = isUniform

