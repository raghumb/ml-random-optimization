import mlrose
import numpy as np

class OneMax:
    def __init__(self, length):
        self.length = length
        pass

    def create_problem(self):
        fitness = mlrose.FlipFlop()
        problem = mlrose.DiscreteOpt(length = self.length, fitness_fn = fitness, maximize = True, max_val = 2)
        return problem