import importlib

# Reloading: remove for production
import cnc.optimization as optimization
importlib.reload(optimization)

import cnc.visualization as visualization
importlib.reload(visualization)

from cnc.optimization import ShortestPath
from cnc.visualization import Visualizer


### Initialize the optimization object
# Path to input file
path_file = './path_files/p1_stpl001.code'

# Size of population per generation
pop_size = 100

# Probability of reproduction
repro = 0.8

# Probability of crossover
crossover = 0.9

# Probability of mutation
mutation = 0.001

# Number of generations to evolve
#  num_generations = 1000
num_generations = 2

# Whether to remember best population, for debugging/visualization
debug = True

opt = ShortestPath(
        path_file, pop_size, repro, crossover, mutation, num_generations,
        debug
        )

### Run the optimization
opt.optimize()

### Plot the results
viz = Visualizer(opt)
viz.visualize_solution()
