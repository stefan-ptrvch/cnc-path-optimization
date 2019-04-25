import importlib

# Reloading: remove for production
import cnc.optimization as optimization
importlib.reload(optimization)

from cnc.optimization import CNCOptimizer

# Path to input file
path_file = './path_files/p3_stpl001.code'

# Generate optimization object
opt = CNCOptimizer(path_file, recipe_grouping=True)

# Run the optimization
opt.optimize()

# Write the optimization to a file
#  opt.save()

# Generate visualization files
opt.visualize()
