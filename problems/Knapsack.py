import mlrose
import numpy as np

class Knapsack:
    def __init__(self, length):
        self.length = length
        pass

    def create_problem(self):
        """weights = [10, 5, 2, 8, 15, 4, 12, 9, 7]
        values =  [1, 2, 3, 4, 5, 6, 7, 8, 9]"""
        weights = [10, 5, 2, 8, 15]
        values =  [1, 2, 3, 4, 5]
        max_weight_pct = 0.6
        fitness = mlrose.Knapsack(weights, values, max_weight_pct)
        problem = mlrose.DiscreteOpt(length = 5, fitness_fn = fitness, maximize = True, max_val = 6)
        return problem