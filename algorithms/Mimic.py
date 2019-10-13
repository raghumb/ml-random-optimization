import mlrose
import numpy as np


class Mimic:

    def __init__(self, problem, random_state = 42,
                max_attempts = 10, max_iters = 1000, pop_size = 200, keep_pct = 0.2):
        self.problem = problem
        self.random_state = random_state
        self.max_attempts = max_attempts
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.keep_pct = keep_pct


    def optimize(self):
        init_state = np.array([0, 1, 0, 1, 1, 1, 1])

        best_state, best_fitness, fitness_curve = mlrose.mimic(
                                     self.problem, 
                                     pop_size = self.pop_size,
                                     keep_pct = self.keep_pct,
                                     max_attempts = self.max_attempts, 
                                     max_iters = self.max_iters, 
                                     curve = True, 
                                     random_state = self.random_state)

        #print('best_state '+ str(best_state))
        #print('best_fitness '+ str(best_fitness))
        #print('fitness_curve '+ str(fitness_curve))
        return best_fitness




                            
        
