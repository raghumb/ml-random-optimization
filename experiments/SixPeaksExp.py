from problems.FlipFlop import FlipFlop
from algorithms.RHC import RHC

class SixPeaksExp:
    def __init__(self):
        pass

    def experiment(self):
        se = SixPeaks(length = 20)
        problem  = se.create_problem()
        rhc_a = RHC(problem)
        best_state ,best_fitness = rhc_a.optimize()
        print(best_fitness)
        print(best_state)



