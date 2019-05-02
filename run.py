import importlib

# Reloading: remove for production
import cnc.optimization as optimization
importlib.reload(optimization)

from cnc.optimization import CNCOptimizer

# Path to input file
path_file = './path_files/test6.code'

# Generate optimization object
opt = CNCOptimizer(path_file, recipe_grouping=True)

# Run the optimization
opt.optimize()

# Write the optimization to a file
opt.save('result')

# Generate visualization file
opt.visualize()
