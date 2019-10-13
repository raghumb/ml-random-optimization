from problems.FlipFlop import FlipFlop
from problems.SixPeaks import SixPeaks
from problems.Knapsack import Knapsack

class CustomProblem:
    def __init__(self, length, problem_type):
        self.length = length
        self.problem_type = problem_type
        pass

    def create_problem(self):
        problem_var = None
        
        if self.problem_type == 'FlipFlop':            
            problem = FlipFlop(self.length)
        elif self.problem_type == '6Peaks':
            problem = SixPeaks(self.length)
        elif self.problem_type == 'Knapsack':
            problem = Knapsack(self.length)            
        
        problem_var = problem.create_problem() 
        return problem_var   