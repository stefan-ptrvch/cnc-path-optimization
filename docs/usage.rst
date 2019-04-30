Usage
*****

CNCOptimizer
============

The *CNCOptimizer* class is the only class needed in order to run the
optimization. An instance of the class gets initialized with the path to the
.code file that needs to be processed, and a flag that tells the optimizer
whether to ignore recipes or not.

After instantiating an *CNCOptimizer* object, the *.optimize* method needs to
be called, after which the *.save* method is used to save the result of the
optimization to a .code file.

A visualization can be generated, if needed, using the *.visualize* method,
which generates a *result.html* file, that can be viewed using any browser.

Full Example
============

A full example, which is also provided with this package, in the *run.py*
file, is given below:

.. code-block:: python

   from cnc.optimization import CNCOptimizer

   # Path to input file
   path_file = './path_files/lea_stpl001_fused.code'

   # Generate optimization object
   opt = CNCOptimizer(path_file, recipe_grouping=False)

   # Run the optimization
   opt.optimize()

   # Write the optimization to a file
   opt.save('result')

   # Generate visualization file
   opt.visualize()
