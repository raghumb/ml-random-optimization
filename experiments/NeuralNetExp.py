import numpy as np
import mlrose
from sklearn import metrics
import WineDataReader as WineDataReader
import DataSplitter as DataSplitter
from experiments.ExperimentHelper import ExperimentHelper
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score, make_scorer

class NeuralNetExp:
    def __init__(self):
        self.initialize()
        pass

    def initialize(self):
        reader = WineDataReader.WineDataReader()
        self.splitter = DataSplitter.DataSplitter(reader)
        self.splitter.read_split_data()

    def experiment(self):

        self.model_complexity_exp_max_attempts_geom_sa()
        self.model_complexity_exp_max_attempts_rhc()
        self.model_complexity_exp_epoch_ga()
        self.model_complexity_exp_max_attempts_rhc()
        self.model_complexity_exp_restart_rhc()
        self.model_complexity_exp_epoch_sa()
        self.model_complexity_exp_popsize_ga()
        self.model_complexity_exp_epoch_ga()
        self.experiment_optimal_rhc()
        self.experiment_optimal_sa()
        self.experiment_optimal_ga()



    def model_complexity_exp(self):
        scoring = 'accuracy'
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [50,50,50,50], activation = 'relu', \
                                 algorithm = 'random_hill_climb', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 5, max_attempts = 100, \
                                 random_state = 3)

        expHelper = ExperimentHelper(self.splitter, nn_model1, 'RHC')
        param_range = np.array([0.0001, 0.001,0.002,0.003,0.005,0.008])
        #param_range = np.array([100, 200,300,400, 500])
        expHelper.model_complexity_exp('learning_rate', param_range)  


    def model_complexity_exp_max_attempts_rhc(self):
        #TODO should we create a new learner object??
        scoring = 'accuracy'
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [50,50,50,50], activation = 'relu', \
                                 algorithm = 'random_hill_climb', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 5, max_attempts = 100, \
                                 restarts = 10, random_state = 3)

        expHelper = ExperimentHelper(self.splitter, nn_model1, 'RHC')
        #param_range = [1, 10, 50, 100, 200, 500]
        param_range = [50, 100, 200, 300, 400]
        expHelper.model_complexity_exp('max_attempts', param_range)

    def model_complexity_exp_restart_rhc(self):
        #TODO should we create a new learner object??
        scoring = 'accuracy'
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [50,50,50,50], activation = 'relu', \
                                 algorithm = 'random_hill_climb', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 5, max_attempts = 100, \
                                 random_state = 3, restarts = 0)

        expHelper = ExperimentHelper(self.splitter, nn_model1, 'RHC')
        param_range = [1, 10, 50, 100, 200, 300, 400, 500]
        expHelper.model_complexity_exp('restarts', param_range)          

    def model_complexity_exp_epoch_sa(self):
        #TODO should we create a new learner object??
        scoring = 'accuracy'
        decay = mlrose.decay.ArithDecay()
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [50,50,50,50], activation = 'relu', \
                                 algorithm = 'simulated_annealing', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 5, max_attempts = 100, \
                                 schedule = decay,
                                 random_state = 3)

        expHelper = ExperimentHelper(self.splitter, nn_model1, 'SA')
        param_range = [1, 10, 50, 100, 200, 500]
        expHelper.model_complexity_exp('max_iters', param_range)

    def model_complexity_exp_max_attempts_sa(self):
        #TODO should we create a new learner object??
        scoring = 'accuracy'
        decay = mlrose.decay.ExpDecay()
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [50,50,50,50], activation = 'relu', \
                                 algorithm = 'simulated_annealing', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 5, max_attempts = 100, \
                                 schedule = decay,
                                 random_state = 3)

        expHelper = ExperimentHelper(self.splitter, nn_model1, 'SA')
        param_range = [20, 50, 100, 200, 300]
        expHelper.model_complexity_exp('max_attempts', param_range) 
        print('completed max attempt sa nn') 

    def model_complexity_exp_max_attempts_geom_sa(self):
        #TODO should we create a new learner object??
        scoring = 'accuracy'
        decay = mlrose.decay.GeomDecay()
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [50,50,50,50], activation = 'relu', \
                                 algorithm = 'simulated_annealing', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 5, max_attempts = 100, \
                                 schedule = decay,
                                 random_state = 3)

        expHelper = ExperimentHelper(self.splitter, nn_model1, 'SA')
        param_range = [20, 50, 100, 200, 300]
        expHelper.model_complexity_exp('max_attempts', param_range) 
        print('completed max attempt sa nn')               


    def model_complexity_exp_epoch_no_stop_sa(self):
        #TODO should we create a new learner object??
        scoring = 'accuracy'
        
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [50,50,50,50], activation = 'relu', \
                                 algorithm = 'simulated_annealing', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = False, clip_max = 5, max_attempts = 100, \
                                 random_state = 3)

        expHelper = ExperimentHelper(self.splitter, nn_model1, 'SA')
        param_range = [1, 10, 50, 100, 200, 500]
        expHelper.model_complexity_exp('max_iters', param_range)        

    def model_complexity_exp_popsize_ga(self):
        #TODO should we create a new learner object??
        scoring = 'accuracy'
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [50,50,50,50], activation = 'relu', \
                                 algorithm = 'genetic_alg', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = False, clip_max = 5, max_attempts = 200, \
                                 random_state = 3)

        expHelper = ExperimentHelper(self.splitter, nn_model1, 'GA')
        param_range = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
        expHelper.model_complexity_exp('pop_size', param_range)                                 


    def model_complexity_exp_epoch_ga(self):
        #TODO should we create a new learner object??
        scoring = 'accuracy'
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [50,50,50,50], activation = 'relu', \
                                 algorithm = 'genetic_alg', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 5, max_attempts = 100, \
                                 pop_size= 2000, random_state = 3)

        expHelper = ExperimentHelper(self.splitter, nn_model1, 'GA')
        param_range = [1, 10, 50, 100, 200, 500, 700, 1000]
        expHelper.model_complexity_exp('max_iters', param_range) 

    def model_complexity_exp_max_attempt_ga(self):
        #TODO should we create a new learner object??
        scoring = 'accuracy'
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [50,50,50,50], activation = 'relu', \
                                 algorithm = 'genetic_alg', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 5, max_attempts = 100, \
                                 pop_size= 2000, random_state = 3)

        expHelper = ExperimentHelper(self.splitter, nn_model1, 'GA')
        param_range = [20, 50, 100, 200, 300]
        expHelper.model_complexity_exp('max_attempts', param_range)         

    
    def experiment_optimal_rhc(self):
        print('NN rhc optimal')
        start = time.time()
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [50,50,50,50], activation = 'relu', \
                                 algorithm = 'random_hill_climb', max_iters = 700, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 5, max_attempts = 200, \
                                 restarts= 20, random_state = 3)

        nn_model1.fit(self.splitter.X_train, self.splitter.y_train)
        y_pred = nn_model1.predict(self.splitter.X_test)
        end = time.time()
        diff = abs(end - start)
        print('time taken for RHC: ' + str(diff))         
        print("Final Accuracy for RHC" +  
                        str(metrics.accuracy_score(self.splitter.y_test, y_pred)))
        print("Confusion matrix for RHC" +  
                        str(confusion_matrix(self.splitter.y_test, y_pred)))       
        print("Recall score for RHC" +  
                        str(recall_score(self.splitter.y_test, y_pred)))  
        print("Precision score for RHC" + 
                        str(precision_score(self.splitter.y_test, y_pred)))
        print('completed optimal rhc nn')   


    def experiment_optimal_sa(self):
        print('NN rhc optimal')
        decay = mlrose.decay.ExpDecay()
        start = time.time()
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [50,50,50,50], activation = 'relu', \
                                 algorithm = 'simulated_annealing', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 5, max_attempts = 200, \
                                 schedule = decay,
                                 restarts= 20, random_state = 3)

        nn_model1.fit(self.splitter.X_train, self.splitter.y_train)
        y_pred = nn_model1.predict(self.splitter.X_test)
        end = time.time()
        diff = abs(end - start)
        print('time taken for SA: ' + str(diff))         
        print("Final Accuracy for SA" +  
                        str(metrics.accuracy_score(self.splitter.y_test, y_pred)))
        print("Confusion matrix for SA" +  
                        str(confusion_matrix(self.splitter.y_test, y_pred)))       
        print("Recall score for SA" +  
                        str(recall_score(self.splitter.y_test, y_pred)))  
        print("Precision score for SA" + 
                        str(precision_score(self.splitter.y_test, y_pred))) 

              
    def experiment_optimal_ga(self):
        print('NN rhc optimal')
        decay = mlrose.decay.ExpDecay()
        start = time.time()
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [50,50,50,50], activation = 'relu', \
                                 algorithm = 'genetic_alg', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 5, max_attempts = 200, \
                                 pop_size= 1200,
                                 restarts= 20, random_state = 3)

        nn_model1.fit(self.splitter.X_train, self.splitter.y_train)
        y_pred = nn_model1.predict(self.splitter.X_test)
        end = time.time()
        diff = abs(end - start)
        print('time taken for GA: ' + str(diff))         
        print("Final Accuracy for GA" +  
                        str(metrics.accuracy_score(self.splitter.y_test, y_pred)))
        print("Confusion matrix for GA" +  
                        str(confusion_matrix(self.splitter.y_test, y_pred)))       
        print("Recall score for GA" +  
                        str(recall_score(self.splitter.y_test, y_pred)))  
        print("Precision score for GA" + 
                        str(precision_score(self.splitter.y_test, y_pred))) 


