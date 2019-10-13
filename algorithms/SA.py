import mlrose
import numpy as np


class SA:

    def __init__(self, problem, init_state, random_state = 42, schedule_var = 0,
                max_attempts = 10, max_iters = 1000):
        self.problem = problem
        self.random_state = random_state
        self.schedule_var = schedule_var
        self.max_attempts = max_attempts
        self.max_iters = max_iters
        self.init_state = init_state

    def optimize(self):
        init_state = np.array([0, 1, 0, 1, 1, 1, 1])
        if self.schedule_var == 0:
            schedule = mlrose.ExpDecay()
        elif self.schedule_var == 1:
            schedule = mlrose.GeomDecay()
        else:
            schedule = mlrose.ArithDecay()

        best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(
                                     self.problem, 
                                     schedule = schedule,
                                     max_attempts = self.max_attempts, 
                                     max_iters = self.max_iters,
                                     init_state = None, #self.init_state, 
                                     curve = True, 
                                     random_state = self.random_state)

        #print('best_state '+ str(best_state))
        #print('best_fitness '+ str(best_fitness))
        #print('fitness_curve '+ str(fitness_curve))
        return best_state, best_fitness




                            
        
