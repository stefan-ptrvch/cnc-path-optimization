# Reloading: remove for production
import importlib

import csv
import copy
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Manager

# Reloading: remove for production
import cnc.visualization as visualization
importlib.reload(visualization)

from cnc.visualization import Visualizer


class CNCOptimizer:
    """
    Solves an instance of the travelling salesman problem, for the CNC machine.

    Finds the shortest cutting tool travel path (SCTTP) using the genetic
    algorithm optimization method, and the best orientation for the lines using
    the hill-climbing algorithm.

    Parameters
    ----------
    file_path : string
        Path to the file containing the lines that need to be ordered.
    time_factor : float
        Scaling factor for the optimization, which increases the population and
        the number of generations of the algorithm.
    num_threads : int
        Number of parallel runs of the optimization. The best one is returned
        as the result.
    recipe_grouping : bool
        Determines whether the recipe numbers should be ignored or not.

    Attributes
    ----------
    lines : list
        Contains all of the lines, as Line objects, after the input file is
        parsed.
    recipe_grouping : bool
        Determines whether the recipe numbers should be ignored or not.
    groups : OrderedDict
        Key-value pairs where the key is the group name and the value is all
        the Line objects that belong to that group.
    group_sizes : dict
        Key-values pairs where the key is the group name and the value is the
        number of Line objects in that group.
    num_genes : int
        Total number of lines, which determines the width of the population
        matrix (the number of genes per individual).
    time_factor : float
        Scaling factor for the optimization, which increases the population and
        the number of generations of the algorithm.
    pop_size_scaler : int
        Scaling factor which is internal to the optimizer. Controls how the
        population size scales with problem difficulty.
    num_generations_scaler : int
        Scaling factor which is internal to the optimizer. Controls how the
        number of generations scales with problem difficulty.
    population : numpy array
        Contains all of the individuals in the optimization. When it gets
        constructed, it respects the specified group order (see `group_lines`
        and `generate_initial_population` method method).
    sub_pops : dict of numpy arrays
        Contains numpy array views of the population, for all the groups.
    next_sub_pops: dict of numpy arrays
        Contains numpy array views of the next generation, for all the groups.
    bi_directional_scaler : int
        Scaling factor which is internal to the optimizer. Controls how the
        number of iterations for hill-climbing scales with problem difficulty.
    num_threads : int
        Number of parallel runs of the optimization. The best one is returned
        as the result.
    pop_size : int
        Number of individuals in the population, or the number of rows in the
        population matrix. Calculated based on problem difficulty and scaling
        factors.
    maximum_distance : float
        The maximum distance between any two lines, in any orientation. Used
        when calculating the fitness of the population.
    prob_mut : float
        Probability of mutation for any individual in the population.
    num_mut : int
        Number of mutations in the population, calculated based on `prob_mut`.
    num_generations : int
        Number of iterations for which to run the genetic algorithm. Calculated
        based on problem difficulty and scaling factors.
    best_result : dict
        Dictionary containing the solution and non-cutting path cost of the
        best individual in the optimization.
    initial_result : dict
        Dictionary containing the solution and non-cutting path cost of one
        individual at the start of the optimization. Used for comparison in the
        visualization.
    path_cost : numpy array
        Vector containing the path cost of every individual in the population.
    fitness : numpy array
        Vector containing the fitness of every individual in the population.
        It's calculated by subtracting the `path_cost` from the maximum path
        cost possible.
    ONES : numpy array
        Matrix of ones, used for constructing a matrix of path costs, when
        doing crossover.
    """

    def __init__(self, file_path, time_factor=1, num_threads=4,
                 recipe_grouping=True):

        # List of lines, which are parsed from input file
        self.lines = []

        # Whether we use recipe grouping or not
        self.recipe_grouping = recipe_grouping

        # Parse the input file
        self.generate_lines_from_file(file_path)

        # Groups of lines and the number of lines in each group
        self.groups = OrderedDict()
        self.group_sizes = {}

        # Generate the groups
        self.group_lines()

        # Number of genes per individual (width of population matrix)
        self.num_genes = len(self.lines)

        # These are the scaling factors for the algorithm parameters (they
        # determine how the algorithm scales with problem difficulty and with
        # the time factor)
        self.time_factor = time_factor
        self.pop_size_scaler = 2
        self.num_generations_scaler = 10
        self.bi_directional_scaler = 1000

        # Number of threads for the optimization
        self.num_threads = num_threads

        # Population size
        self.pop_size = time_factor*self.pop_size_scaler*self.num_genes

        # Calculate the distance matrix (distance between every two lines in
        # every orientation).
        self.generate_distance_matrix()

        # Get the maximum distance of the distance matrix, used when
        # calculating population fitness from population path cost
        self.max_distance = self.distance_matrix.max()

        # Percentage of mutation
        self.prob_mut = 0.001

        # Number of mutations, derived from percentage
        self.num_mut = int(np.ceil(self.prob_mut*self.pop_size*self.num_genes))

        # Number of generations
        self.num_generations = time_factor*self.num_generations_scaler*self.num_genes

        # Used for storing numpy array of all the individuals
        self.population = None

        # Dictionary containing numpy array views for every group in the
        # population
        self.sub_pops = {}

        # Dictionary containing numpy array views for every group for the next
        # generation
        self.next_sub_pops = {}

        # Best result overall
        self.best_result = {'solution': [], 'path_cost': np.inf}

        # Initial result, only used when plotting the results
        self.initial_result = {'solution': [], 'path_cost': np.inf}

        # Path cost of generation
        self.path_cost = None

        # Fitness of generation (positive value based on path cost)
        self.fitness = None

        # Used in crossover
        self.ONES = np.ones(self.pop_size)

    def generate_lines_from_file(self, file_path):
        """
        Parses the input file generates Line objects for every line, and stores
        them in a list.

        Parameters
        ----------
        file_path : str
            Path to the file containing the lines that need to be optimized.
        """

        # Line ID is simply a counter
        line_id = 0

        # We parse the file now
        with open(file_path, 'r') as path_file:
            reader = csv.reader(path_file, delimiter=' ')
            for row in reader:

                # The name of the line
                line_type = row[0]

                # The number of the recipe
                recipe = row[-1]

                # The coordinates in the file are Y1, X1, Y2, X2, and we
                # want to store them as X1, Y1, X2, Y2
                # We skip the 1st entry since it contains the type of line
                starting_point = np.array([
                    float(row[2].replace(',', '')),
                    float(row[1].replace(',', '')),
                    ])

                endpoint = np.array([
                    float(row[4].replace(',', '')),
                    float(row[3].replace(',', '')),
                    ])

                line = Line(line_type, starting_point, endpoint, recipe,
                            line_id)

                # If it's an EDGEDEL_LINE line type, set the thickness as well
                if line_type == 'EDGEDEL_LINE':
                    line.set_thickness(float(row[5].replace(',', '')))

                # Add the line to the list of lines
                self.lines.append(line)

                # Increment line ID
                line_id += 1

    def group_lines(self):
        """
        Groups Line objects, and stores the group in an ordered dictionary.

        The lines are stored in an ordered dictionary, since we need the groups
        to be in the correct order, according to the following requirements:

        We have to respect the following order:
        1) REF
        2) SCRIBE_LINE (non 2 recipe)
        3) BUSBAR_LINE
        4) EDGEDEL_LINE
        5) SCRIBE_LINE2

        We need this order when iterating over the dict, while generating the
        initial population (see `generate_initial_population` method).
        """

        # First we store the groups in an unordered dict, and later sort them
        # accordingly
        unordered_groups = {}
        for line in self.lines:

            # Determine the line type and recipe number of the line
            line_type = line.get_line_type()
            recipe = line.get_recipe()

            # Determine to which group this line belongs
            if line_type == 'SCRIBE_LINE' and recipe == '2':
                group_name = 'SCRIBE_LINE2'
            elif line_type == 'REF':
                group_name = 'REF'
            elif self.recipe_grouping:
                group_name = line_type
            else:
                group_name = line_type + recipe

            # Add the line to the group
            if group_name not in unordered_groups:
                unordered_groups[group_name] = [line]
            else:
                unordered_groups[group_name].append(line)

        # We have to respect the following order:
        # 1) REF
        # 2) SCRIBE_LINE (non 2 recipe)
        # 3) BUSBAR_LINE
        # 4) EDGEDEL_LINE
        # 5) SCRIBE_LINE2
        # So we now add the lines to a new dict, which is an OrderedDict
        # (respects insertion order, which as of Python 3.7 is a language
        # feature (it's enabled by default))

        # The REF line is always first in the solution, if there is one
        if 'REF' in unordered_groups:
            self.groups['REF'] = unordered_groups['REF']

        # Find all SCRIBE_LINE (non 2 recipe) groups
        for group_name, group in unordered_groups.items():
            if 'SCRIBE_LINE' in group_name and '2' not in group_name:
                self.groups[group_name] = group

        # Find all BUSBAR_LINE groups
        for group_name, group in unordered_groups.items():
            if 'BUSBAR_LINE' in group_name:
                self.groups[group_name] = group

        # Find all EDGEDEL_LINE groups
        for group_name, group in unordered_groups.items():
            if 'EDGEDEL_LINE' in group_name:
                self.groups[group_name] = group

        # Find all SCRIBE_LINE2 groups
        for group_name, group in unordered_groups.items():
            if 'SCRIBE_LINE2' == group_name:
                self.groups[group_name] = group

    def generate_distance_matrix(self):
        """
        Generates matrix of Euclidian distances between every two nodes.

        The distances are generated for every pair of lines, in any
        orientation, meaning that for every two lines there are four distances.
        So the matrix is of 2Nx2N dimensions, where N is the number of lines.
        """

        self.distance_matrix = np.zeros((2*self.num_genes, 2*self.num_genes))

        # Non-flipped lines
        for i in range(self.num_genes):
            for j in range(self.num_genes):
                self.distance_matrix[i, j] = np.linalg.norm(
                    self.lines[i].get_endpoint() - self.lines[j].get_starting_point()
                        )

        # 1st line flipped
        for i in range(self.num_genes):
            for j in range(self.num_genes):
                self.distance_matrix[self.num_genes + i, j] = np.linalg.norm(
                    self.lines[i].get_starting_point() - self.lines[j].get_starting_point()
                        )

        # 2nd line flipped
        for i in range(self.num_genes):
            for j in range(self.num_genes):
                self.distance_matrix[i, self.num_genes + j] = np.linalg.norm(
                    self.lines[i].get_endpoint() - self.lines[j].get_endpoint()
                        )

        # Both lines flipped
        for i in range(self.num_genes):
            for j in range(self.num_genes):
                self.distance_matrix[self.num_genes + i, self.num_genes + j] = np.linalg.norm(
                    self.lines[i].get_starting_point() - self.lines[j].get_endpoint()
                        )

    def evaluate_generation(self):
        """
        Evaluates the path cost and hence fitness of the whole generation,
        using a heuristic.

        The path cost is calculated using the Euclidean distance between every
        two successive lines in an individual. The heuristic that is used tells
        the evaluation to treat every so as if it can be oriented both ways
        simultaneously, so we're actually trying to find the lowest "possible"
        path cost (not the real path cost). We later use hill-climbing to find
        the best orientation for every line so that we get the real lowest path
        cost.
        """

        # Row and column indices of entries in the distance matrix, needed for
        # path cost calculation (see distance matrix)
        rows = self.population[:, :-1]
        cols = self.population[:, 1:]

        # Use a heuristic to try and estimate the possible lowest path cost
        # when accounting for bi-directional optimization
        # We estimate the cost for every possible orientation between every two
        # successive lines in the individual, and then we take the minimum
        # across all four for every line.
        all_costs = np.empty((4, self.pop_size, self.num_genes - 1))
        all_costs[0] = self.distance_matrix[rows, cols]
        all_costs[1] = self.distance_matrix[rows + self.num_genes, cols]
        all_costs[2] = self.distance_matrix[rows, cols + self.num_genes]
        all_costs[3] = self.distance_matrix[
            rows + self.num_genes,
            cols + self.num_genes
        ]

        all_costs = all_costs.min(axis=0)

        # Get the cost estimate
        self.path_cost = all_costs.sum(axis=1)

        # Calculate the fitness
        self.fitness = self.num_genes*self.max_distance - self.path_cost

    def crossover(self):
        """
        Generates the next generation using crossover. The group order is kept
        intact while doing this.

        Takes two individuals at a time, using a roulette game based on
        fitness, and crosses them using the Order 1 Crossover method. There are
        some clever numpy tricks used in this method (for performance reasons),
        which are not very verbose/intuitive, so I included a lot of commentary
        above these lines.
        """

        # Perform crossover for all sub-populations
        for group_name, group in self.sub_pops.items():
            if group_name == 'REF':
                continue

            # We're playing roulette, so we have to generate a ball that falls
            # on some individual (we take a bunch of individuals from the
            # population).
            # We have the cumulative sum of the fitness function of the
            # population, so a fit individual will have a big slice of the
            # cumulative sum. If we then generate a random number between 0 and
            # the maximum of the cumulative sum, individuals with a bigger
            # slice will have a higher probability to be selected, meaning
            # fitter individuals will have a higher probability to pass on
            # their genes.
            # In order to find the individual on which "the ball fell", we ask
            # what is the index of the 1st individual whose cumulative fitness
            # value is greater than the value of the "ball". Some individuals
            # may be selected multiple times, while other may not be selected
            # at all.

            # Generate the balls
            balls = np.ceil(
                np.random.uniform(size=self.pop_size)*self.cumulative[-1]
            )

            # Find the indices of the winners of the roulette game. The idea is
            # to find the individual on whose cumulative sum "slice" the ball
            # fell. We do array comparison, and find the 1st index that is
            # greater than the value of the ball. We do this by generating a
            # matrix where every row has the cumulative fitness (same vector)
            # and we compare it with the randomly generated balls that fall on
            # some individual.
            indices_of_winners = np.argmax(
                (np.outer(self.ONES, self.cumulative).T > balls).T,
                axis=1
            )

            # Now we select all the individuals who won the roulette game
            parents = group[indices_of_winners, :]

            # Arrays to be populated with genetic material (the children)
            child1 = np.empty(self.group_sizes[group_name])
            child2 = np.empty(self.group_sizes[group_name])

            # Unfortunately, I couldn't think of a way to do the crossover part
            # without a loop, but at least it's more readable this way
            for i in np.arange(0, self.pop_size, 2):
                parent1 = parents[i]
                parent2 = parents[i + 1]

                # We're doing Order 1 crossover

                # Generate points, used for cutting out genetic material
                crp1 = np.random.randint(self.group_sizes[group_name])
                crp2 = np.random.randint(crp1, self.group_sizes[group_name])

                # Populate children with a cut of material
                child1[crp1:crp2] = parent1[crp1:crp2]
                child2[crp1:crp2] = parent2[crp1:crp2]

                # Fill in rest of material using Order 1 crossover
                other_genes_child1 = list(set(parent2) - set(parent1[crp1:crp2]))
                other_genes_child2 = list(set(parent1) - set(parent2[crp1:crp2]))
                group_size = self.group_sizes[group_name]
                for gen_num in range(group_size - (crp2 - crp1)):
                    idx = (crp2 + gen_num) % group_size
                    child1[idx] = other_genes_child1[gen_num]
                    child2[idx] = other_genes_child2[gen_num]

                # Add the children to the population
                self.next_sub_pops[group_name][i, :] = child1
                self.next_sub_pops[group_name][i + 1, :] = child2

    def mutation(self):
        """
        Performs mutation by swapping two genes around. The group order is kept
        intact while doing this.
        """

        # Go through all the groups, and perform the wanted number of mutations
        for group_name, group in self.next_sub_pops.items():
            if group_name == 'REF':
                continue

            # Perform the mutations
            for i in range(self.num_mut):
                individual = int(np.ceil(np.random.uniform()*self.pop_size - 1))
                gene1 = int(np.ceil(np.random.uniform()*self.group_sizes[group_name] - 1))
                gene2 = int(np.ceil(np.random.uniform()*self.group_sizes[group_name] - 1))
                gene = group[individual, gene1]
                group[individual, gene1] = group[individual, gene2]
                group[individual, gene2] = gene

    def generate_initial_population(self):
        """
        Generates initial population, while also implementing correct group
        order, and constructing views that point to slices of the population
        that represent every group.
        """

        # The global population contains the sub-populations, and optimization
        # is being done on the sub-population level, but the path cost is
        # considered at the global population level
        self.population = np.ndarray((self.pop_size, self.num_genes)).astype(int)

        # The sub-populations are views of the population matrix
        cols_populated = 0
        for group_name, group in self.groups.items():

            # Remember the number of lines in the group
            self.group_sizes[group_name] = len(group)

            # Get all the IDs of the lines in this group, which will be used
            # when performing optimization
            group_line_numbers = []
            for line in group:
                group_line_numbers.append(line.get_line_id())

            self.sub_pops[group_name] = self.population[
                :,
                cols_populated:cols_populated + self.group_sizes[group_name]
            ]
            cols_populated += self.group_sizes[group_name]

            # Generate the initial state for the sub-population
            for i in range(self.pop_size):
                self.sub_pops[group_name][i] = np.random.permutation(group_line_numbers)

    def optimize(self):
        """
        Runs the specified number of threads, each doing a complete path and
        direction optimization with a different random seed. Stores the best
        result of all the optimization runs.
        """

        # List containing all optimization processes
        processes = []

        # Manager which will take care of shared state (all the optimization
        # objects)
        process_manager = Manager()
        best_list = process_manager.list()
        initial_list = process_manager.list()

        # Start a thread for every group
        progress_bar_position = 0
        for i in range(self.num_threads):

            # Create the processes
            p = Process(target=self.opt_thread, args=(best_list,
                                                      initial_list,
                                                      progress_bar_position
                                                      )
                        )
            processes.append(p)
            p.start()
            progress_bar_position += 1

        # Wait for processes to finish before executing other code
        for process in processes:
            process.join()

        # Take an initial population, only used for visualization
        self.initial_result = initial_list[0]

        # Now find the best solution
        best_solution_cost = np.inf
        for solution in best_list:
            if solution['path_cost'] < best_solution_cost:
                self.best_result = solution
                best_solution_cost = solution['path_cost']

    def opt_thread(self, best_list, initial_list, progress_bar_position):
        """
        Gets launched as a child process by the main process and performs one
        optimization, with a newly generated random seed.

        Parameters
        ----------
        best_list : process manager list
            List that is shared between all threads, to which the result of the
            optimization gets appended to. Result gets appended after
            orientation optimization.
        initial_list : process manager list
            List that is shared between all threads, to which an individual
            from the start of the optimization gets appended to (used for
            visualization purposes).
        progress_bar_position : int
            Determines the row in which the tqdm progress bar gets drawn. Also
            determines part of the thread name.
        """

        # Set the seed for this thread, since it has the same seed as the
        # parent thread
        np.random.seed()

        self.generate_initial_population()

        # Number of iterations of one run
        for generation in tqdm(
            range(self.num_generations),
            desc="Path Optimization " + str(progress_bar_position),
            position=progress_bar_position
        ):

            # Evaluate the current generation
            self.evaluate_generation()

            # Get the best individual and their path cost
            if self.best_result['path_cost'] > self.path_cost.min():
                self.best_result['solution'] = self.population[
                        self.path_cost.argmin()
                        ].copy()
                self.best_result['path_cost'] = self.path_cost.min()

                # If this is the 1st iteration, save the result for comparison
                if generation < 1:
                    self.initial_result['solution'] = self.population[0].copy()
                    self.initial_result['path_cost'] = self.path_cost[0].copy()

            # Calculate cumulative sum for roulette game (used for
            # reproduction and crossover)
            self.cumulative = self.fitness.cumsum()

            # Generate placeholders for the next generation sub-populations
            for group in self.sub_pops:
                self.next_sub_pops[group] = np.empty((
                    self.pop_size,
                    self.group_sizes[group]
                ))
                if group == 'REF':
                    self.next_sub_pops[group][:] = self.sub_pops[group][:]

            # Perform crossover
            self.crossover()

            # Perform mutation
            self.mutation()

            # Migrate population
            for group_name, group in self.next_sub_pops.items():
                self.sub_pops[group_name][:] = group[:]

        # Append initial result to list of initial results (only used for
        # visualization)
        initial_list.append(self.initial_result)

        # Find the best orientation for the lines in the result, using the
        # hill-climbing algorithm
        self.bi_directional(best_list, progress_bar_position)

    def bi_directional(self, best_list, progress_bar_position):
        """
        Uses hill-climbing to find the best orientation for every line, so that
        it minimizes the path cost.

        Parameters
        ----------
        best_list : process manager list
            List that is shared between all threads, to which the result of the
            optimization gets appended to.
        progress_bar_position : int
            Determines the row in which the tqdm progress bar gets drawn. Also
            determines part of the thread name.
        """

        # Take the non-bi-directional solution as the current best solution,
        # and calculate its real path cost (not possible lowest, which is the
        # heuristic see `evaluate_generation` method)
        result = self.best_result['solution']
        current_best = result[:]
        rows = current_best[:-1]
        cols = current_best[1:]
        current_best_cost = self.distance_matrix[rows, cols].sum()
        current_best_flip = np.zeros(self.num_genes, dtype=bool)

        num_iter = self.num_genes*self.bi_directional_scaler*self.time_factor
        for i in tqdm(
            np.arange(num_iter),
            desc="Direction Optimization " + str(progress_bar_position),
            position=self.num_threads + progress_bar_position
        ):
            # Generate a vector which determines which of the lines need to be
            # flipped
            flip = np.zeros((self.num_genes), dtype='int')
            flip[np.random.uniform(size=self.num_genes) > i/num_iter] = self.num_genes

            # Index into distance matrix, and calculate path cost
            bi_result = result + flip
            rows = bi_result[:-1]
            cols = bi_result[1:]
            cost = self.distance_matrix[rows, cols].sum()
            if cost < current_best_cost:
                current_best_cost = cost
                current_best = bi_result[:]
                current_best_flip = flip

        # Set the bi-directionally optimized result as the best result
        self.best_result['path_cost'] = current_best_cost

        # Remember which lines need to be flipped
        self.best_result['flip'] = current_best_flip > 0

        # Append the result to the list that is shared between threads
        best_list.append(self.best_result)

    def get_result(self):
        """
        Returns the correctly flipped, optimized order of Line objects.

        Returns
        -------
        out : list of Line
            Correctly flipped, optimized order of Line objects
        -------
        """

        # Create copies of the lines, so as not to change the lines stored
        # after parsing
        copied_lines = [copy.deepcopy(line) for line in self.lines]

        # Flip lines that need to be flipped
        for index, value in enumerate(self.best_result['solution']):
            if self.best_result['flip'][index]:
                copied_lines[value].flip_line()

        # Order the lines correctly
        return [copied_lines[index] for index in self.best_result['solution']]

    def get_initial(self):
        """
        Returns order of Line object obtained at the beginning of optimization.

        Returns
        -------
        out : list of Line
            Lines ordered in a random fashion, from an individual at the
            beginning of optimization.
        """

        return [self.lines[index] for index in self.initial_result['solution']]

    def visualize(self):
        """
        Visualizes the result of the optimization, using the Visualizer class.
        """

        viz = Visualizer(self.get_result(), self.get_initial())
        viz.visualize()

    def save(self, file_name):
        """
        Saves the results of the optimization to a file. Adds .code file
        extension if it's not specified.

        Parameters
        ----------
        file_name : str
            Filename of the file which will contain the optimized result.
        """

        # Add file extension if there's none
        if '.code' not in file_name:
            file_name += '.code'

        # Open file
        with open(file_name, 'w') as f:

            # Write the lines to the new file
            for line in self.get_result():

                # The format is:
                # LINE_TYPE, Y1, X1, Y2, X2, [TH], RE
                formatted = line.get_line_type()
                formatted += ' '
                formatted += "{:.3f}".format(line.get_starting_point()[1])
                formatted += ', '
                formatted += "{:.3f}".format(line.get_starting_point()[0])
                formatted += ', '
                formatted += "{:.3f}".format(line.get_endpoint()[1])
                formatted += ', '
                formatted += "{:.3f}".format(line.get_endpoint()[0])
                formatted += ', '

                # If it's and EDGEDEL_LINE line type, add the thickness as well
                if 'EDGEDEL_LINE' == line.get_line_type():
                    formatted += "{:.3f}".format(line.get_thickness())
                    formatted += ', '

                formatted += line.get_recipe()
                formatted += '\n'

                f.write(formatted)


class Line:
    """
    Line which represents where the CNC head will perform cutting.

    Parameters
    ----------
    line_type : str
        Name of the line type, contained in the 1st column of the .code
        file.
    starting_point : np.array
        Numpy array of two coordinates, X1 and Y1, representing the
        starting point of cutting.
    endpoint : np.array
        Numpy array of two coordinates, X2 and Y2, representing the
        endpoint of cutting.
    recipe : str
        Recipe number, last column of .code file.
    line_id : int
        Unique line ID, used when constructing the initial population, so as to
        order the lines correctly (to respect the group order).
    flip : bool
        Determines whether the starting and endpoints need to be flipped or not.
    """

    def __init__(self, line_type, starting_point, endpoint, recipe, line_id, flip=False):

        self.line_type = line_type
        self.starting_point = starting_point
        self.endpoint = endpoint
        self.recipe = recipe
        self.line_id = line_id

        # Determines whether the starting and endpoints should be flipped or not
        self.flip = flip

    def set_thickness(self, thickness):
        """
        Set the line thickness, which is only specified for EDGEDEL_LINE line
        types.

        Parameters
        ----------
        thickness : str
            Number representing the thinkess of the line. Not used for
            calculations, only when writing to new .code file.
        """

        self.thickness = thickness

    def get_line_type(self):
        """
        Returns the type of line.

        Returns
        -------
        line_type : str
            Name of line type.
        """

        return self.line_type

    def get_starting_point(self):
        """
        Returns the starting point of a line.

        Returns
        -------
        out : np.array
            Numpy array of two coordinates, X1 and Y1, representing the
            starting point of cutting. Returns endpoint if lines are flipped.
        """

        if not self.flip:
            return self.starting_point
        else:
            return self.endpoint

    def get_endpoint(self):
        """
        Returns the endpoint of a line.

        Returns
        -------
        out : np.array
            Numpy array of two coordinates, X2 and Y2, representing the
            endpoint of cutting. Returns the starting point if lines are
            flipped.
        """

        if not self.flip:
            return self.endpoint
        else:
            return self.starting_point

    def get_recipe(self):
        """
        Returns the recipe of a line.

        Returns
        -------
        recipe : str
            Recipe number.
        """

        return self.recipe

    def get_thickness(self):
        """
        Returns thickness of EDGEDEL_LINE line type.

        Returns
        -------
        thickness : str
            Number representing the thickness of the line.
        """

        return self.thickness

    def get_line_id(self):
        """
        Returns the ID of the line.

        Returns
        -------
        line_id : int
            Number representing the ID of the line, which was given to it while
            parsing the input file.
        """

        return self.line_id

    def flip_line(self):
        """
        Flips the starting end endpoints.
        """

        self.flip = True
