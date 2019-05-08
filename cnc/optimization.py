# Reloading: remove for production
import importlib

import csv
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

# Reloading: remove for production
import cnc.visualization as visualization
importlib.reload(visualization)

from cnc.visualization import Visualizer


class CNCOptimizer():
    """
    Solves an instance of the travelling salesman problem, for the CNC machine.

    Finds the shortest cutting tool travel path (SCTTP) using the genetic
    algorithm optimization method.
    """

    def __init__(self, file_path, pop_size, repro, crossover, mutation, num_generations, recipe_grouping=True):
        """
        Parameters
        ----------
        nodes : list of Line
            List of Line objects which are used to generate the distance
            matrix.
        pop_size : int
            Size of the population per generation.
        repro : float
            Probability of reproduction.
        crossover : float
            Probability of crossover.
        mutation : float
            Probability of mutation
        num_generations : int
            Number of iterations to run the algorithm for.
        progress_bar_position : int
            Row in which the progress bar should be displayed.
        """

        # List of lines, which are parsed from input file
        self.lines = []

        # Whether we use recipe grouping or not
        self.recipe_grouping = recipe_grouping

        # Parse the input file
        self.generate_lines_from_file(file_path)

        # Groups of lines
        self.groups = OrderedDict()
        self.group_sizes = {}

        # Generate the groups
        self.group_lines()

        # Number of genes per individual
        self.num_genes = len(self.lines)

        # Population size
        self.pop_size = pop_size

        # Get the distance matrix for the nodes
        self.generate_distance_matrix()

        # Get the maximum distance
        self.max_distance = self.distance_matrix.max()

        # Probability of reproduction
        self.prob_repro = repro

        # Percentage of crossover
        self.prob_cross = crossover

        # Set of nodes, used for fixing children
        self.set_of_nodes = set(np.arange(self.num_genes))

        # Percentage of mutation
        self.prob_mut = mutation

        # Calculate how many reproductions and crossovers we need per
        # generation, as well as how many mutations we're gonna do
        self.num_repro = np.round((1 - self.prob_repro)*self.pop_size).astype(int)
        self.num_cross = self.pop_size - self.num_repro
        self.num_mut = int(np.ceil(self.prob_mut*self.pop_size*self.num_genes))

        # We need an even number of crossovers (because we need pairs)
        if np.mod(self.num_cross, 2) == 1:
            self.num_cross += 1
            self.num_repro -= 1

        # Number of generations
        self.num_generations = num_generations

        # Best result overall
        self.best_result = {'solution': [], 'path_cost': np.inf}

        # Initial result
        self.initial_result = {'solution': [], 'path_cost': np.inf}

        # Path cost of generation
        self.path_cost = None

        # Fitness of generation, which is inverse normalized path cost (for
        # optimization puproses)
        self.fitness = None

    def generate_lines_from_file(self, file_path):
        """
        Parses the input file and generates Line objects for every line.
        """

        # Line ID is simply a counter
        line_id = 0
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

                # Generate the node
                line = Line(line_type, starting_point, endpoint, recipe,
                        line_id)

                # If it's an EDGEDEL_LINE line type, set the thikness as well
                if line_type == 'EDGEDEL_LINE':
                    line.set_thikness(float(row[5].replace(',', '')))

                # Add the line to the list of lines
                self.lines.append(line)

                # Increment line ID
                line_id += 1

    def group_lines(self):
        """
        Groups Line objects.
        """

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

        The matrix is of 2Nx2N size, where N is the number of genes, since the
        line can be oriented either way.
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
        Evaluates the path cost and fitness of the whole generation.

        Path cost is calculated as the Euclidian distance between the second
        poin in a node and the first point in the next node. Fitness is
        calculated as the maxmimum possible path cost, minus the actual path
        cost.
        """

        # Row and column indices of entries distance matrix, needed for path
        # cost calculation (see distance matrix)
        rows = self.population[:, :-1]
        cols = self.population[:, 1:]

        # Use a heuristic to try and estimate the possible lowest path cost
        # when accounting for bi-directional optimization
        all_costs = np.empty((4, self.pop_size, self.num_genes - 1))
        all_costs[0] = self.distance_matrix[rows, cols]
        all_costs[1] = self.distance_matrix[rows + self.num_genes, cols]
        all_costs[2] = self.distance_matrix[rows, cols + self.num_genes]
        all_costs[3] = self.distance_matrix[
                rows + self.num_genes,
                cols + self.num_genes
                ]

        all_costs = all_costs.min(axis=0)

        # Get the cost
        #  self.path_cost = self.distance_matrix[rows, cols].sum(axis=1)
        self.path_cost = all_costs.sum(axis=1)

        # Calculate the fitness
        self.fitness = self.num_genes*self.max_distance - self.path_cost

    def reproduction(self):
        """
        Determines which individuals get to move to the next generation (which
        ones get cloned).
        """

        # Perform reproduction for all sub-populations
        for group_name, group in self.sub_pops.items():
            if group_name == 'REF':
                continue

            # We're playing roulette, so we have to generate a ball that falls on
            # some individual
            for i in range(self.num_repro):

                # Generate the ball
                ball = np.ceil(np.random.uniform()*self.cumulative[-1]).astype(int)

                # Take the winner of the roulette game, and clone him into the next
                # generation
                index_of_winner = np.argmax(self.cumulative > ball).astype(int)
                self.next_sub_pops[group_name][i, :] = group[index_of_winner, :]

    def crossover(self):
        """
        Generates part of the population using crossover.

        Takes two individuals at a time, based on fitness and combines them,
        using the Order 1 Crossover method.
        """

        # Perform crossover for all sub-populations
        for group_name, group in self.sub_pops.items():
            if group_name == 'REF':
                continue

            # We're playing roulette, so we have to generate a ball that falls on
            # some individual
            for i in range(self.num_cross//2):

                # Generate the ball
                ball = np.ceil(np.random.uniform()*self.cumulative[-1]).astype(int)

                # Take the winner of the roulette game
                index_of_winner = np.argmax(self.cumulative > ball).astype(int)
                parent1 = group[index_of_winner, :]

                # Do the whole thing again for the second parent
                ball = np.ceil(np.random.uniform()*self.cumulative[-1]).astype(int)
                index_of_winner = np.argmax(self.cumulative > ball).astype(int)
                parent2 = group[index_of_winner, :]

                # We're doing Odrder 1 crossover
                if np.random.uniform() < self.prob_cross:
                    # Generate points, used for cutting out genetic material
                    crp1 = np.random.randint(self.group_sizes[group_name])
                    crp2 = np.random.randint(crp1, self.group_sizes[group_name])

                    # Arrays to be populated with genetic material
                    child1 = np.empty(self.group_sizes[group_name])
                    child2 = np.empty(self.group_sizes[group_name])

                    # Populate children with a cut of material
                    child1[crp1:crp2] = parent1[crp1:crp2]
                    child2[crp1:crp2] = parent2[crp1:crp2]

                    # Fill in rest of material using Order 1 crossover
                    other_genes_child1 = list(set(parent2) - set(parent1[crp1:crp2]))
                    other_genes_child2 = list(set(parent1) - set(parent2[crp1:crp2]))
                    for gen_num in range(self.group_sizes[group_name] - (crp2 - crp1)):
                        idx = (crp2 + gen_num) % self.group_sizes[group_name]
                        child1[idx] = other_genes_child1[gen_num]
                        child2[idx] = other_genes_child2[gen_num]

                else:
                    child1 = parent1[:]
                    child2 = parent2[:]

                # Add the children to the population
                self.next_sub_pops[group_name][self.num_repro + 2*i, :] = child1
                self.next_sub_pops[group_name][self.num_repro + 2*i + 1, :] = child2

    def mutation(self):
        """
        Mutates a set number of individuals in the population, by swapping two
        genes.
        """

        for group_name, group in self.next_sub_pops.items():
            if group_name == 'REF':
                continue
            for i in range(self.num_mut):
                individual = int(np.ceil(np.random.uniform()*self.pop_size - 1))
                gene1 = int(np.ceil(np.random.uniform()*self.group_sizes[group_name] - 1))
                gene2 = int(np.ceil(np.random.uniform()*self.group_sizes[group_name] - 1))
                gene = group[individual, gene1]
                group[individual, gene1] = group[individual, gene2]
                group[individual, gene2] = gene

    def generate_initial_population(self):
        """
        Generates initial population.
        """

        # The global population contains the sub-populations, and optimization
        # is being done on the sub-population level, but the path cost is
        # considered at the global population level
        self.population = np.ndarray((self.pop_size, self.num_genes)).astype(int)

        # The sub-populations are views of the population matrix
        self.sub_pops = {}
        cols_populated = 0
        for group_name, group in self.groups.items():

            # Remember the number of lines in the gorup
            self.group_sizes[group_name] = len(group)

            # Get all the IDs of the lines in this group, which will be used
            # when performing optimization
            group_line_numbers = []
            for line in group:
                group_line_numbers.append(line.get_line_id())

            self.sub_pops[group_name] = self.population[
                    :,
                    cols_populated: cols_populated + self.group_sizes[group_name]
                    ]
            cols_populated += self.group_sizes[group_name]

            # Generate the initial state for the sub-population
            for i in range(self.pop_size):
                self.sub_pops[group_name][i] = np.random.permutation(group_line_numbers)

    def optimize(self):
        """
        Runs the optimization algorithm trying to find the shortest path.
        """

        self.generate_initial_population()

        # Number of iterations of one run
        for generation in tqdm(
                range(self.num_generations),
                desc="Path Optimization"
                ):

            # Evaluate the current generation
            self.evaluate_generation()

            ### DEBUG
            #  print(generation, self.path_cost.mean())
            ### DEBUG

            # Get the best individual and his path cost
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
            self.next_sub_pops = {}
            for group in self.sub_pops:
                self.next_sub_pops[group] = np.empty((
                    self.pop_size,
                    self.group_sizes[group]
                    ))
                if group == 'REF':
                    self.next_sub_pops[group][:] = self.sub_pops[group][:]

            # Perform reproduction
            self.reproduction()

            # Perform crossover
            self.crossover()

            # Perform mutation
            self.mutation()

            # Migrate population
            for group_name, group in self.next_sub_pops.items():
                self.sub_pops[group_name][:] = group[:]

        # Try finding the best orientation for all the lines, using the
        # hill-descent algorithm
        print(self.best_result)
        self.bi_directional()

    def bi_directional(self):
        """
        Uses hill-descent to find the best solution for bi-directional
        problem.
        """

        # Take the path cost of the non-bi-directional solution as the
        # currenlty best solution
        result = self.best_result['solution']
        current_best = result[:]
        rows = current_best[:-1]
        cols = current_best[1:]
        current_best_cost = self.distance_matrix[rows, cols].sum()
        current_best_flip = np.zeros(self.num_genes, dtype=bool)

        num_iter = self.num_genes*10000
        for i in range(num_iter):
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
                print("FOUND:", current_best_cost)

        # Set the bi-directionally optimized result as the best
        self.best_result['path_cost'] = current_best_cost
        self.best_result['flip'] = current_best_flip > 0

    def get_result(self):
        """

        """
        # Flip lines that need to be flipped
        for index, value in enumerate(self.best_result['solution']):
            if self.best_result['flip'][index]:
                self.lines[value].flip_line()

        return [self.lines[index] for index in self.best_result['solution']]

    def get_initial(self):
        """

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
        if not '.code' in file_name:
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
                # If it's and EDGEDEL_LINE line type, add the thikness as well
                if 'EDGEDEL_LINE' == line.get_line_type():
                    formatted += "{:.3f}".format(line.get_thikness())
                    formatted += ', '
                formatted += line.get_recipe()
                formatted += '\n'

                f.write(formatted)


class Line():
    """
    Line which represents where the CNC head will perform cutting.
    """

    def __init__(self, line_type, starting_point, endpoint, recipe, line_id):
        """
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
        """

        self.line_type = line_type
        self.starting_point = starting_point
        self.endpoint = endpoint
        self.recipe = recipe
        self.line_id = line_id

        # Determines whether the starting and endpoins should be flipped or not
        self.flip = False

    def set_thikness(self, thikness):
        """
        Set the line thikness, which is only specified for EDGEDEL_LINE line
        types.

        Parameters
        ----------
        thikness : str
            Number representing the thinkess of the line. Not used for
            calculations, only when writing to new .code file.
        """

        self.thikness = thikness

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
        starting_point : np.array
            Numpy array of two coordinates, X1 and Y1, representing the
            starting point of cutting.
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
        endpoint : np.array
            Numpy array of two coordinates, X2 and Y2, representing the
            endpoint of cutting.
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

    def get_thikness(self):
        """
        Returns thikness of EDGEDEL_LINE line type.

        Returns
        -------
        thikness : str
            Number representing the thikness of the line.
        """

        return self.thikness

    def get_line_id(self):
        """
        Returns the ID of the line
        """

        return self.line_id

    def flip_line(self):
        """
        Flips the starting end endpoins.
        """

        self.flip = True
