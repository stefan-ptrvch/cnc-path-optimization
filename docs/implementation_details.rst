Implementation Details
**********************

The problem has two requirements. One is that a certain line group order needs
to be respected, and the other one is that lines can be cut either way.

How Group Order Is Preserved
============================

The line order that needs to be respected is the following one:

   #. REF
   #. SCRIBE_LINE (non 2 recipe)
   #. BUSBAR_LINE
   #. EDGEDEL_LINE
   #. SCRIBE_LINE2

In order for this grouping to be respected, the lines are grouped together
after the .code file is parsed, and when the population matrix is being
initialized, line groups are placed into the matrix left to right, column-wise,
respecting the specified group order, using an ordered dictionary (standard in
Python 3.7, but implemented as a special *OrderedDict* class in earlier
versions).

While initializing the population matrix, pointers to parts of the population
matrix are being constructed (so called numpy array views). So, one can access
and manipulate every group of the population individually, without affecting
the group ordering that was specified during the initialization of the
population matrix. Hence, in the *crossover* and *mutation* methods, only these
groups are used to performs these actions, so there's only inner-group mixing
of genetic material (lines can't get out of their respective groups).

On the other hand, the path cost (and fitness) gets calculated "globally", that
is, using the whole population matrix, not individually for every group. This
ensures that individuals with best inner-group line ordering are favored.

How It's Optimized
==================
There are two steps to the optimization. The first step is to find the best
possible line order, taking into account that the lines can be oriented either
way. This is done using the genetic algorithm with a heuristic instead of a
precise fitness function. The heuristic tries to estimate the lowest possible
path cost, if all the lines can be simultaneously oriented both ways, which
means, that the heuristic gives better scores to line orders which can
*potentially* have a very low path cost, if the right line orientation is
found.

The second step is to find the best orientation for every line, for the line
order that was determined by the genetic algorithm using the heuristic. This is
done using the *hill-climbing* algorithm. While performing this part of the
optimization, the real path cost is used instead of the heuristic.
