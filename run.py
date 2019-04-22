import importlib

# Reloading: remove for production
import cnc.optimization as optimization
importlib.reload(optimization)

import cnc.visualization as visualization
importlib.reload(visualization)

from cnc.optimization import CNCOptimizer

# Path to input file
path_file = './path_files/p3_stpl001.code'

# Generate optimization object
opt = CNCOptimizer(path_file, recipe_grouping=False)

# Run the optimization
opt.optimize()

# Generate visualization files
opt.visualize()
