import importlib

# Reloading: remove for production
import cnc.optimization as optimization
importlib.reload(optimization)

from cnc.optimization import ShortestPath



### Initialize the optimization object
# Path to input file
path_file = './path_files/p3_stpl001.code'

# Size of population per generation
pop_size = 50

# Probability of reproduction
repro = 0.8

# Probability of crossover
crossover = 0.9

# Probability of mutation
mutation = 0.001

# Number of generations to evolve
num_generations = 100

# Number of optimizational runs to perform
num_runs = 1

opt = ShortestPath(
        path_file, pop_size, repro, crossover, mutation, num_generations,
        num_runs
        )

### Run the optimization
opt.optimize()

### Plot the results
