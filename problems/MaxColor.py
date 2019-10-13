import mlrose
import numpy as np

class MaxColor:
    def __init__(self, length):
        self.length = length
        pass

    def create_problem(self):
        edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        fitness = mlrose.FlipFlop()
        problem = mlrose.DiscreteOpt(length = self.length, fitness_fn = fitness, maximize = True, max_val = 2)
        return problem