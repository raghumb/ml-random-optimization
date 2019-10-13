import mlrose
import numpy as np

class SixPeaks:
    def __init__(self, length, t_pct = 0.1):
        self.t_pct = t_pct
        self.length = length

    def create_problem(self):
        fitness = mlrose.SixPeaks(t_pct = self.t_pct)
        problem = mlrose.DiscreteOpt(length = self.length, fitness_fn = fitness, maximize = True, max_val = 2)
        return problem