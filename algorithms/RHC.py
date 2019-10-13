import mlrose
import numpy as np


class RHC:

    def __init__(self, problem, init_state, random_state = 42, max_attempts = 10, max_iters = 1000, restarts =0):
        self.problem = problem
        self.random_state = random_state
        self.max_attempts = max_attempts
        self.max_iters = max_iters
        self.init_state = init_state 
        self.restarts = restarts       

    def optimize(self):
        
        best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(self.problem, 
                                     max_attempts = self.max_attempts, 
                                     max_iters = self.max_iters, 
                                     restarts = self.restarts,
                                     init_state = None, #self.init_state, 
                                     curve = True, 
                                     random_state = self.random_state)

        #print('best_state '+ str(best_state))
        #print('best_fitness '+ str(best_fitness))
        #print('fitness_curve '+ str(fitness_curve))
        return best_state, best_fitness




                            
        
