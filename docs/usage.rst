Usage
*****

CNCOptimizer
============

The *CNCOptimizer* class is the only class needed in order to run the
optimization. An instance of the class gets initialized with the path to the
.code file that needs to be processed, and a flag that tells the optimizer
whether to ignore recipes or not. Two optinal arguments are available, that can
be used to fine tune the scale of the optimization. The arguments are
*time_factor*, which increases the number of iterations and the population size
of the optimization, and *num_threads*, which controls the number of parallel
optimizations that are run. The optimizations all have a different random seed,
so they all find a slightly different solution, and the best one is taken as
the result. The altorithm also automatically scales with problem difficulty
(the number of lines), and the ammount of scaling is also controlled with the
*time_factor* argument.

Visualization now takes longer than optimization.
After instantiating an *CNCOptimizer* object, the *.optimize* method needs to
be called, after which the *.save* method is used to save the result of the
optimization to a .code file.

A visualization can be generated, if needed, using the *.visualize* method,
which generates a *visualization.html* file, that can be viewed using any
browser.

Full Example
============

A full example, which is also provided with this package, in the *run.py*
file, is given below:

.. code-block:: python

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
