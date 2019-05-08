import importlib

# Reloading: remove for production
import cnc.optimization as optimization
importlib.reload(optimization)

from cnc.optimization import CNCOptimizer

# Path to input file
path_file = './path_files/p3_stpl001.code'

# Number of generations
num_generations = 1000

# Size of population
pop_size = 500

# Probability of reproduction
repro = 0.8

# Probability of crossover
crossover = 0.9

# Probability of mutation
mutation = 0.001

# Generate optimization object
opt = CNCOptimizer(path_file, pop_size, repro, crossover, mutation,
        num_generations, recipe_grouping=True)

# Run the optimization
opt.optimize()

# Write the optimization to a file
opt.save('result')

# Generate visualization file
opt.visualize()
