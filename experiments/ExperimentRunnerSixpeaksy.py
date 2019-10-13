from problems.CustomProblem import CustomProblem
from problems.SixPeaks import SixPeaks
from algorithms.RHC import RHC
from algorithms.SA import SA
from algorithms.GA import GA
from algorithms.Mimic import Mimic
from experiments.ExperimentHelper import ExperimentHelper
import numpy as np
from plotter import plot_curve
import time

class ExperimentRunnerSixpeaksy:
    def __init__(self, problem_type):
        self.expHelper = ExperimentHelper()
        self.problem_type = problem_type
        self.rand_seeds = self.expHelper.create_random_seeds()
        pass

    def experiment(self):
        self.experiment_rhc_2()
        self.experiment_rhc_3()

        
        self.experiment_sa_2()
        self.experiment_sa_3()
        self.experiment_sa_4()
        self.experiment_sa_5()
        self.experiment_sa_6()       
        self.experiment_sa_7()                          

        #self.experiment_ga_1()
        self.experiment_ga_2()
        self.experiment_ga_3() 
        self.experiment_ga_4() 
        self.experiment_ga_5() 

        #self.experiment_mimic_1()
        self.experiment_mimic_2()
        self.experiment_mimic_3()
        self.experiment_mimic_4()
        self.experiment_mimic_5()

        self.experiment_sa()
        self.experiment_ga()
        self.experiment_mimc()
        self.experiment_optimal_rhc()
        self.experiment_optimal_sa()
        self.experiment_optimal_ga()
        self.experiment_optimal_mimic()


    def experiment_rhc_1(self):
        init_state = None
        restart_lengths = np.arange(10, 400, 50)
        result = np.zeros((len(self.rand_seeds), len(restart_lengths)))
        #best_state = np.zeros((len(self.rand_seeds), len(restart_lengths)))
        print(self.problem_type)
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            prob_length = 20
            for j in range(len(restart_lengths)):
                restart_length = restart_lengths[j]
                max_iter = np.inf
                #max_attempts is varied by trial and error 
                max_attempts = 10
                fl = CustomProblem(prob_length, self.problem_type)
                problem  = fl.create_problem()
                alg = RHC(problem, init_state, rand_state, max_attempts, max_iter, restart_length.item())
                best_state, best_fitness = alg.optimize()
                result[i][j] = best_fitness
                

        print('best fitness')
        print(str(result))
        print('best state')
        print(best_state)
        avg_result = np.mean(result, axis = 0)
        print('avg result for varying input size'+ str(avg_result))
        title = self.problem_type + ' with RHC - # of Restarts Variation'
        plot_curve(restart_lengths, avg_result, title, '# of Restarts', 'Best Score')


    def experiment_rhc_22(self):
        print('in 22')
        init_state = None
        max_attempts = np.array([5, 10, 15, 30, 40, 50,60, 80, 100, 200])
        result = np.zeros((len(self.rand_seeds), len(max_attempts)))
        best_score = None
        for i in range(len(self.rand_seeds)):
            restarts = 0
            rand_state = self.rand_seeds[i]
            for j in range(len(max_attempts)):
                prob_length = 20
                fl = CustomProblem(prob_length, self.problem_type)
                problem  = fl.create_problem()
                max_attempt = max_attempts[j].item()
                max_iter = np.inf
                alg = RHC(problem, init_state, rand_state, max_attempt, max_iter, restarts)
                best_score, best_fitness = alg.optimize()
                result[i][j] = best_fitness

        avg_result = np.mean(result, axis = 0)
        print('avg result '+ str(avg_result))
        print('best score')
        print(best_score)
        title = self.problem_type + ' with RHC - Max Attempts Variation - 0 restarts'
        plot_curve(max_attempts, avg_result, title, 'Max Attempts', 'Best Score')




    def experiment_rhc_11(self):
        init_state = None
        prob_lengths = np.arange(7, 30)
        result = np.zeros((len(self.rand_seeds), len(prob_lengths)))
        print(self.problem_type)
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(prob_lengths)):
                prob_length = prob_lengths[j]
                fl = CustomProblem(prob_length.item(), self.problem_type)
                problem  = fl.create_problem()
                alg = RHC(problem, init_state, rand_state, 10, 1000)
                best_fitness = alg.optimize()
                result[i][j] = best_fitness

        print(str(result))
        avg_result = np.mean(result, axis = 0)
        print('avg result for varying input size'+ str(avg_result))
        title = self.problem_type + ' with RHC - Input Size Variation'
        plot_curve(prob_lengths, avg_result, title, 'Input Size', 'Best Score')


    def experiment_rhc_2(self):
        init_state = None
        max_attempts = np.array([5, 10, 15, 30, 40, 50,60, 80, 100, 200])
        result = np.zeros((len(self.rand_seeds), len(max_attempts)))
        best_score = None
        for i in range(len(self.rand_seeds)):
            restarts = 20
            rand_state = self.rand_seeds[i]
            for j in range(len(max_attempts)):
                prob_length = 20
                fl = CustomProblem(prob_length, self.problem_type)
                problem  = fl.create_problem()
                max_attempt = max_attempts[j].item()
                max_iter = np.inf
                alg = RHC(problem, init_state, rand_state, max_attempt, max_iter, restarts)
                best_score, best_fitness = alg.optimize()
                result[i][j] = best_fitness

        avg_result = np.mean(result, axis = 0)
        print('avg result '+ str(avg_result))
        print('best score')
        print(best_score)
        title = self.problem_type + ' with RHC - Max Attempts Variation'
        plot_curve(max_attempts, avg_result, title, 'Max Attempts', 'Best Score')


    def experiment_rhc_3(self):
        init_state = None
        max_iters = np.arange(100, 5000, 100)
        result = np.zeros((len(self.rand_seeds), len(max_iters)))
        
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(max_iters)):
                prob_length = 20
                fl = CustomProblem(prob_length, self.problem_type)
                problem  = fl.create_problem()
                max_attempt = 50
                restarts = 20
                max_iter = max_iters[j].item()
                alg = RHC(problem, init_state, rand_state, max_attempt, max_iter, restarts)
                best_score, best_fitness = alg.optimize()
                result[i][j] = best_fitness

        avg_result = np.mean(result, axis = 0)
        print('avg result '+ str(avg_result))
        print('best score')
        print(best_score)
        title = self.problem_type + ' with RHC - Max Iterations Variation'
        plot_curve(max_iters, avg_result, title, 'Max Iterations', 'Best Score')


    def experiment_rhc_4(self):
        init_state = None
        t_pcts = np.arange(0.1, 1, 0.1)
        result = np.zeros((len(self.rand_seeds), len(t_pcts)))
        best_score = None
        max_iter = np.inf
        for i in range(len(self.rand_seeds)):
            restarts = 400
            rand_state = self.rand_seeds[i]
            for j in range(len(t_pcts)):
                prob_length = 20
                restarts = 400
                max_attempt = 50
                t_pct = t_pcts[j].item()
                fl = SixPeaks(prob_length, t_pct)
                problem  = fl.create_problem()
                
                alg = RHC(problem, init_state, rand_state, max_attempt, max_iter, restarts)
                best_score, best_fitness = alg.optimize()
                result[i][j] = best_fitness

        avg_result = np.mean(result, axis = 0)
        print('avg result '+ str(avg_result))
        print('best score')
        print(best_score)
        title = self.problem_type + ' with RHC - Threshold Variation'
        plot_curve(t_pcts, avg_result, title, 'Threshold', 'Best Score')


##################################################################
########### SA ###################################
#SA(problem, 42, 0, 10, 1000)
    def experiment_sa_11(self):
        init_state = None
        prob_lengths = np.arange(7, 30)
        schedule_var = 0
        best_state = None
        result = np.zeros((len(self.rand_seeds), len(prob_lengths)))
        best_state = None
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(prob_lengths)):
                prob_length = prob_lengths[j]
                fl = CustomProblem(prob_length.item(), self.problem_type)
                problem  = fl.create_problem()
                alg = SA(problem, init_state, rand_state, schedule_var, 10, 1000)
                best_state,best_fitness = alg.optimize()
                result[i][j] = best_fitness

        print(str(result))
        print('best_state')
        print(best_state)        
        avg_result = np.mean(result, axis = 0)
        print('avg result for varying input size'+ str(avg_result))
        title = self.problem_type + ' with SA - Input Size Variation'
        plot_curve(prob_lengths, avg_result, title, 'Input Size', 'Best Score')



    def experiment_sa_22(self):
        init_state = None
        schedule_var = 0
        best_state = None
        max_attempts = np.arange(50, 60, 5)
        print(max_attempts)
        result = np.zeros((len(self.rand_seeds), len(max_attempts)))
        
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(max_attempts)):
                prob_length = 20
                fl = CustomProblem(prob_length, self.problem_type)
                problem  = fl.create_problem()
                max_attempt = max_attempts[j].item()
                max_iter = np.inf
                alg = SA(problem, init_state, rand_state, schedule_var, max_attempt, max_iter)
                best_state, best_fitness = alg.optimize()
                result[i][j] = best_fitness

        avg_result = np.mean(result, axis = 0)
        print('avg result '+ str(avg_result))
        print('best_state')
        print(best_state)        
        title = self.problem_type + ' with SA - Max Attempts Variation'
        plot_curve(max_attempts, avg_result, title, 'Max Attempts', 'Best Score')



    def experiment_sa_2(self):
        init_state = None
        schedule_var = 0
        best_state = None
        max_attempts = np.array([5, 10, 15, 30, 40, 50,60, 80, 100]) #np.arange(100, 600, 100)
        result = np.zeros((len(self.rand_seeds), len(max_attempts)))
        
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(max_attempts)):
                prob_length = 20
                
                fl = CustomProblem(prob_length, self.problem_type)
                problem  = fl.create_problem()
                max_attempt = max_attempts[j].item()
                max_iter = np.inf
                alg = SA(problem, init_state, rand_state, schedule_var, max_attempt, max_iter)
                best_state, best_fitness = alg.optimize()
                result[i][j] = best_fitness

        avg_result = np.mean(result, axis = 0)
        print('avg result '+ str(avg_result))
        print('best_state')
        print(best_state)        
        title = self.problem_type + ' with SA - Max Attempts Variation-Exp Decay'
        plot_curve(max_attempts, avg_result, title, 'Max Attempts', 'Best Score')


    def experiment_sa_3(self):
        init_state = None
        schedule_var = 0
        best_state = None
        max_iters = np.arange(100, 5000, 100)
        result = np.zeros((len(self.rand_seeds), len(max_iters)))
        
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(max_iters)):
                prob_length = 20
                fl = CustomProblem(prob_length, self.problem_type)
                problem  = fl.create_problem()
                
                max_iter = max_iters[j].item()
                max_attempt = 70
                alg = SA(problem, init_state, rand_state, schedule_var, max_attempt, max_iter)
                best_state, best_fitness = alg.optimize()
                result[i][j] = best_fitness

        avg_result = np.mean(result, axis = 0)
        print('best_state')
        print(best_state)        
        print('avg result '+ str(avg_result))
        title = self.problem_type + ' with SA - Max Iterations Variation-Exp Decay'
        plot_curve(max_iters, avg_result, title, 'Max Iterations', 'Best Score')

    def experiment_sa_4(self):
        init_state = None
        schedule_var = 1
        best_state = None
        max_attempts = np.array([5, 10, 15, 30, 40, 50,60, 80, 100])
        result = np.zeros((len(self.rand_seeds), len(max_attempts)))
        
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(max_attempts)):
                prob_length = 20
                max_iter = np.inf
                fl = CustomProblem(prob_length, self.problem_type)
                problem  = fl.create_problem()
                max_attempt = max_attempts[j].item()
                alg = SA(problem, init_state, rand_state, schedule_var, max_attempt, max_iter)
                best_state, best_fitness = alg.optimize()
                result[i][j] = best_fitness

        avg_result = np.mean(result, axis = 0)
        print('avg result '+ str(avg_result))
        print('best_state')
        print(best_state)        
        title = self.problem_type + ' with SA - Max Attempts Variation -Geom'
        plot_curve(max_attempts, avg_result, title, 'Max Attempts', 'Best Score')        


    def experiment_sa_5(self):
        init_state = None
        schedule_var = 1
        best_state = None
        max_iters = np.arange(100, 5000, 100)
        result = np.zeros((len(self.rand_seeds), len(max_iters)))
        
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(max_iters)):
                prob_length = 20
                fl = CustomProblem(prob_length, self.problem_type)
                problem  = fl.create_problem()
                max_attempt = 90
                max_iter = max_iters[j].item()
                alg = SA(problem, init_state, rand_state, schedule_var, max_attempt, max_iter)
                best_state, best_fitness = alg.optimize()
                result[i][j] = best_fitness

        avg_result = np.mean(result, axis = 0)
        print('best_state')
        print(best_state)        
        print('avg result '+ str(avg_result))
        title = self.problem_type + ' with SA - Max Iter Variation - Geom'
        plot_curve(max_iters, avg_result, title, 'Max Iterations', 'Best Score')


    def experiment_sa_6(self):
        init_state = None
        schedule_var = 2
        best_state = None
        max_attempts = np.array([5, 10, 15, 30, 40, 50,60, 80, 100])
        result = np.zeros((len(self.rand_seeds), len(max_attempts)))
        
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(max_attempts)):
                prob_length = 20
                max_iter = np.inf
                fl = CustomProblem(prob_length, self.problem_type)
                problem  = fl.create_problem()
                max_attempt = max_attempts[j].item()
                alg = SA(problem, init_state, rand_state, schedule_var, max_attempt, max_iter)
                best_state, best_fitness = alg.optimize()
                result[i][j] = best_fitness

        avg_result = np.mean(result, axis = 0)
        print('avg result '+ str(avg_result))
        print('best_state')
        print(best_state)        
        title = self.problem_type + ' with SA - Max Attempts Variation -Arith'
        plot_curve(max_attempts, avg_result, title, 'Max Attempts', 'Best Score')        


    def experiment_sa_7(self):
        init_state = None
        schedule_var = 2
        best_state = None
        max_iters = np.arange(100, 10000, 100)
        result = np.zeros((len(self.rand_seeds), len(max_iters)))
        
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(max_iters)):
                prob_length = 20
                fl = CustomProblem(prob_length, self.problem_type)
                problem  = fl.create_problem()
                max_attempt = 90
                max_iter = max_iters[j].item()
                alg = SA(problem, init_state, rand_state, schedule_var, max_attempt, max_iter)
                best_state, best_fitness = alg.optimize()
                result[i][j] = best_fitness

        avg_result = np.mean(result, axis = 0)
        print('best_state')
        print(best_state)        
        print('avg result '+ str(avg_result))
        title = self.problem_type + ' with SA - Max Iter Variation - Arith'
        plot_curve(max_iters, avg_result, title, 'Max Iterations', 'Best Score')

#####################################################################
# ##  GA ############################
# GA(problem, 42, 10, 1000, 200, 0.1)

    def experiment_ga_1(self):        
        prob_lengths = np.arange(7, 30)        
        result = np.zeros((len(self.rand_seeds), len(prob_lengths)))
        pop_size = 200
        mutation_prob = 0.1
        
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(prob_lengths)):
                prob_length = prob_lengths[j]
                fl = CustomProblem(prob_length.item(), self.problem_type)
                problem  = fl.create_problem()
                alg = GA(problem, rand_state, 10, 1000, pop_size, mutation_prob)
                best_fitness = alg.optimize()
                result[i][j] = best_fitness

        print(str(result))
        avg_result = np.mean(result, axis = 0)
        print('avg result for varying input size'+ str(avg_result))
        title = self.problem_type + ' with GA - Input Size Variation'
        plot_curve(prob_lengths, avg_result, title, 'Input Size', 'Best Score')


    def experiment_ga_2(self):
        max_attempts = np.array([5, 10, 15, 30, 40, 50,60, 80, 100])
        result = np.zeros((len(self.rand_seeds), len(max_attempts)))
        pop_size = 200
        mutation_prob = 0.1
        
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(max_attempts)):
                prob_length = 20
                fl = CustomProblem(prob_length, self.problem_type)
                problem  = fl.create_problem()
                max_attempt = max_attempts[j].item()
                max_iter = np.inf
                alg = GA(problem, rand_state, max_attempt, max_iter, pop_size, mutation_prob)
                best_fitness = alg.optimize()
                result[i][j] = best_fitness

        avg_result = np.mean(result, axis = 0)
        print('avg result '+ str(avg_result))
        title = self.problem_type + ' with GA - Max Attempts Variation'
        plot_curve(max_attempts, avg_result, title, 'Max Attempts', 'Best Score')


    def experiment_ga_3(self):
        max_iters = np.arange(1000, 5000, 100)
        result = np.zeros((len(self.rand_seeds), len(max_iters)))
        pop_size = 1200
        mutation_prob = 0.1
        
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(max_iters)):
                prob_length = 20
                fl = CustomProblem(prob_length, self.problem_type)
                problem  = fl.create_problem()
                max_attempt = 100
                max_iter = max_iters[j].item()
                alg = GA(problem, rand_state, max_attempt, max_iter, pop_size, mutation_prob)
                best_fitness = alg.optimize()
                result[i][j] = best_fitness

        avg_result = np.mean(result, axis = 0)
        print('avg result '+ str(avg_result))
        title = self.problem_type + ' with GA - Max Iterations Variation'
        plot_curve(max_iters, avg_result, title, 'Max Iterations', 'Best Score')


    def experiment_ga_4(self):
        pop_sizes = np.arange(200, 2000, 200)
        result = np.zeros((len(self.rand_seeds), len(pop_sizes)))
        mutation_prob = 0.3
        max_iter = np.inf
        
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(pop_sizes)):
                prob_length = 20
                fl = CustomProblem(prob_length, self.problem_type)
                problem  = fl.create_problem()
                max_attempt = 100
                pop_size = pop_sizes[j].item()
                alg = GA(problem, rand_state, max_attempt, max_iter, pop_size, mutation_prob)
                best_fitness = alg.optimize()
                result[i][j] = best_fitness

        avg_result = np.mean(result, axis = 0)
        print('avg result '+ str(avg_result))
        title = self.problem_type + ' with GA - Population Size Variation'
        plot_curve(pop_sizes, avg_result, title, 'Population Size', 'Best Score')


    def experiment_ga_5(self):
        mutation_probs = np.arange(0.1, 1, 0.1)
        result = np.zeros((len(self.rand_seeds), len(mutation_probs)))
        pop_size = 3500
        max_iter = np.inf
        
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(mutation_probs)):
                prob_length = 20
                fl = CustomProblem(prob_length, self.problem_type)
                problem  = fl.create_problem()
                max_attempt = 80
                mutation_prob = mutation_probs[j].item()
                alg = GA(problem, rand_state, max_attempt, max_iter, pop_size, mutation_prob)
                best_fitness = alg.optimize()
                result[i][j] = best_fitness

        avg_result = np.mean(result, axis = 0)
        print('avg result '+ str(avg_result))
        title = self.problem_type + ' with GA - Mutation Prob Variation'
        plot_curve(mutation_probs, avg_result, title, 'Mutation Prob', 'Best Score')


#####################################################################
# ##  mimic ############################
# Mimic(problem, 42, 10, 1000, 200, 0.1)

    def experiment_mimic_1(self):        
        prob_lengths = np.arange(7, 30)        
        result = np.zeros((len(self.rand_seeds), len(prob_lengths)))
        pop_size = 200
        keep_pct = 0.1
        
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(prob_lengths)):
                prob_length = prob_lengths[j]
                fl = CustomProblem(prob_length.item(), self.problem_type)
                problem  = fl.create_problem()
                alg = Mimic(problem, rand_state, 10, 1000, pop_size, keep_pct)
                best_fitness = alg.optimize()
                result[i][j] = best_fitness

        print(str(result))
        avg_result = np.mean(result, axis = 0)
        print('avg result for varying input size'+ str(avg_result))
        title = self.problem_type + ' with mimic - Input Size Variation'
        plot_curve(prob_lengths, avg_result, title, 'Input Size', 'Best Score')


    def experiment_mimic_2(self):
        max_attempts = np.array([5, 10, 15, 30, 40, 50,60, 80, 100])
        result = np.zeros((len(self.rand_seeds), len(max_attempts)))
        pop_size = 200
        keep_pct = 0.1
        max_iter = np.inf
        
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(max_attempts)):
                prob_length = 20
                fl = CustomProblem(prob_length, self.problem_type)
                problem  = fl.create_problem()
                max_attempt = max_attempts[j].item()
                alg = Mimic(problem, rand_state, max_attempt, max_iter, pop_size, keep_pct)
                best_fitness = alg.optimize()
                result[i][j] = best_fitness

        avg_result = np.mean(result, axis = 0)
        print('avg result '+ str(avg_result))
        title = self.problem_type + ' with Mimic - Max Attempts Variation'
        plot_curve(max_attempts, avg_result, title, 'Max Attempts', 'Best Score')


    def experiment_mimic_3(self):
        max_iters = np.arange(1000, 5000, 100)
        result = np.zeros((len(self.rand_seeds), len(max_iters)))
        pop_size = 800
        keep_pct = 0.1
        
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(max_iters)):
                prob_length = 20
                fl = CustomProblem(prob_length, self.problem_type)
                problem  = fl.create_problem()
                max_attempt = 100
                max_iter = max_iters[j].item()
                alg = Mimic(problem, rand_state, max_attempt, max_iter, pop_size, keep_pct)
                best_fitness = alg.optimize()
                result[i][j] = best_fitness

        avg_result = np.mean(result, axis = 0)
        print('avg result '+ str(avg_result))
        title = self.problem_type + ' with Mimic - Max Iterations Variation'
        plot_curve(max_iters, avg_result, title, 'Max Iterations', 'Best Score')

    def experiment_mimic_4(self):
        pop_sizes = np.arange(200, 1000, 200)
        result = np.zeros((len(self.rand_seeds), len(pop_sizes)))
        max_iter = np.inf
        keep_pct = 0.1
        
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(pop_sizes)):
                prob_length = 20
                fl = CustomProblem(prob_length, self.problem_type)
                problem  = fl.create_problem()
                max_attempt = 100
                pop_size = pop_sizes[j].item()
                alg = Mimic(problem, rand_state, max_attempt, max_iter, pop_size, keep_pct)
                best_fitness = alg.optimize()
                result[i][j] = best_fitness

        avg_result = np.mean(result, axis = 0)
        print('avg result '+ str(avg_result))
        title = self.problem_type + ' with Mimic - Population Size Variation'
        plot_curve(pop_sizes, avg_result, title, 'Population Size', 'Best Score')

    def experiment_mimic_5(self):
        keep_pcts = np.arange(0.1, 1, 0.1)
        result = np.zeros((len(self.rand_seeds), len(keep_pcts)))
        pop_size = 500
        max_iter = np.inf
        
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]
            for j in range(len(keep_pcts)):
                prob_length = 20
                fl = CustomProblem(prob_length, self.problem_type)
                problem  = fl.create_problem()
                max_attempt = 70
                keep_pct = keep_pcts[j].item()
                alg = Mimic(problem, rand_state, max_attempt, max_iter, pop_size, keep_pct)
                best_fitness = alg.optimize()
                result[i][j] = best_fitness

        avg_result = np.mean(result, axis = 0)
        print('avg result '+ str(avg_result))
        title = self.problem_type + ' with Mimic - Keep PCT Variation'
        plot_curve(keep_pcts, avg_result, title, 'Keep PCT', 'Best Score')        

    def experiment_sa(self):
        fl = FlipFlop(7)
        problem  = fl.create_problem()
        alg = SA(problem, 42, 0, 10, 1000)
        alg.optimize() 

    def experiment_ga(self):
        fl = FlipFlop(7)
        problem  = fl.create_problem()
        alg = GA(problem, 42, 10, 1000, 200, 0.1)
        alg.optimize() 

    def experiment_mimc(self):
        fl = FlipFlop(7)
        problem  = fl.create_problem()
        alg = Mimic(problem, 42, 10, 1000, 200, 0.1)
        alg.optimize()                        


    def experiment_optimal_rhc(self):
        prob_length = 20
        max_attempt = 50
        restarts = 20
        max_iter = 1000
        rand_state = 42
        init_state = None
        fl = CustomProblem(prob_length, self.problem_type)
        problem  = fl.create_problem()
        start = time.time()
        alg = RHC(problem, init_state, rand_state, max_attempt, max_iter, restarts)
        best_score, best_fitness = alg.optimize()
        end = time.time()
        diff = abs(end - start)
        print('time taken for RHC- Sixpeaks: ' + str(diff))

    def experiment_optimal_sa(self):
        prob_length = 20
        init_state = None
        schedule_var = 0
        rand_state = 42
        max_attempt = 80
        fl = CustomProblem(prob_length, self.problem_type)
        problem  = fl.create_problem()
        start = time.time()
        alg = SA(problem, init_state, rand_state, schedule_var, max_attempt, 1000)
        best_score, best_fitness = alg.optimize()
        end = time.time()
        diff = abs(end - start)
        print('time taken for SA - Sixpeaks: ' + str(diff))     


    def experiment_optimal_ga(self):
        prob_length = 20
        fl = CustomProblem(prob_length, self.problem_type)
        problem  = fl.create_problem()
        pop_size = 1200
        rand_state = 42
        max_attempt = 100   
        max_iter = 1000     
        mutation_prob = 0.1
        start = time.time()
        alg = GA(problem, rand_state, max_attempt, max_iter, pop_size, mutation_prob)
        best_fitness = alg.optimize()
        end = time.time()
        diff = abs(end - start)
        print('time taken for GA- Sixpeaks: ' + str(diff)) 


    def experiment_optimal_mimic(self):
        prob_length = 20
        fl = CustomProblem(prob_length, self.problem_type)
        problem  = fl.create_problem()
        pop_size = 800
        rand_state = 42
        max_attempt =100   
        max_iter = 1000     
        keep_pct = 0.1        
        start = time.time()
        alg = Mimic(problem, rand_state, max_attempt, max_iter, pop_size, keep_pct)
        best_fitness = alg.optimize()
        end = time.time()
        diff = abs(end - start)
        print('time taken for Mimic - Sixpeaks: ' + str(diff))  