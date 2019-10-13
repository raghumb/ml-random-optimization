import mlrose
import numpy as np


class GA:

    def __init__(self, problem, random_state = 42,
                max_attempts = 10, max_iters = 1000, pop_size = 200, mutation_prob = 0.1):
        self.problem = problem
        self.random_state = random_state
        self.max_attempts = max_attempts
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob


    def optimize(self):
        

        best_state, best_fitness, fitness_curve = mlrose.genetic_alg(
                                     self.problem, 
                                     pop_size = self.pop_size,
                                     mutation_prob = self.mutation_prob,
                                     max_attempts = self.max_attempts, 
                                     max_iters = self.max_iters,
                                     curve = True, 
                                     random_state = self.random_state)

        #print('best_state '+ str(best_state))
        #print('best_fitness '+ str(best_fitness))
        #print('fitness_curve '+ str(fitness_curve))
        return best_fitness




                            
        
