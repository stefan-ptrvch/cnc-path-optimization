import importlib

# Reloading: remove for production
import cnc.optimization as optimization
importlib.reload(optimization)

from cnc.optimization import CNCOptimizer

# Path to input file
path_file = './path_files/p3_stpl001.code'

# Scaling factor which determines the size of the optimization (length and
# scale) and hence the time as well
timing_factor = 1

# Number of threads to run for the optimization (default is 4)
num_threads = 4

# Generate optimization object
opt = CNCOptimizer(path_file, timing_factor, num_threads, recipe_grouping=True)

# Run the optimization
opt.optimize()

# Write the optimization to a file
opt.save('result')

# Generate visualization file
opt.visualize()
