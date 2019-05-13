import importlib

# Reloading: remove for production
import cnc.optimization as optimization
importlib.reload(optimization)

from cnc.optimization import CNCOptimizer

# Path to input file
PATH_FILE = './path_files/p3_stpl001.code'

# Number of generations
NUM_GENERATIONS = 1000

# Size of population
POP_SIZE = 500

# Probability of reproduction
REPRO = 0.8

# Probability of crossover
CROSSOVER = 0.9

# Probability of mutation
MUTATION = 0.001

# Generate optimization object
OPT = CNCOptimizer(PATH_FILE, POP_SIZE, REPRO, CROSSOVER, MUTATION,
                   NUM_GENERATIONS, recipe_grouping=True)

# Run the optimization
OPT.optimize()

# Write the optimization to a file
OPT.save('result')

# Generate visualization file
OPT.visualize()
