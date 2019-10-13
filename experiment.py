from experiments.ExperimentRunner import ExperimentRunner
from experiments.ExperimentRunnerKnapsack import ExperimentRunnerKnapsack
from experiments.NeuralNetExp import NeuralNetExp
from experiments.ExperimentRunnerSixpeaksy import ExperimentRunnerSixpeaksy

def experiment():


    er = ExperimentRunner('FlipFlop')
    er.experiment()    



    er = ExperimentRunnerKnapsack('Knapsack')
    er.experiment() 
    

    nn = NeuralNetExp()
    nn.experiment()

    er = ExperimentRunnerSixpeaksy('6Peaks')
    er.experiment()




experiment()