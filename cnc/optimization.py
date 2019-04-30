# Reloading: remove for production
import importlib

import csv
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Manager, current_process

# Reloading: remove for production
import cnc.visualization as visualization
importlib.reload(visualization)

from cnc.visualization import Visualizer


class CNCOptimizer():
    """
    Finds the shortest path for CNC cutting.

    Parses the input file, and generates a number of GeneticAlgorithm objects,
    for different groups of lines, and performs the optimization in parallel.
    Makes visualizations of the cutting trajectory using bokeh.
    """

    def __init__(self, file_path, recipe_grouping=True):
        """
        Parameters
        ----------
        file_path : str
            The path to a .code file which needs to be optimized.
        recipe_grouping : bool
            Tells the optimizer whether to treat recipes individually or group
            them together.
        """

        # Path to file which contains line coordinates which need to be
        # optimized
        self.file_path = file_path

        # Whether to do recipe grouping
        self.recipe_grouping = recipe_grouping

        # List of all lines
        self.lines = []

        # Reference line
        self.ref_line = None

        # List holding the result
        self.result = []

        # List holding the initial order of lines
        self.initial = []

        # Generate dictionary of lines
        self.generate_lines_from_file()

    def generate_lines_from_file(self):
        """
        Parses the input file and generates Line objects for every line.
        """

        with open(self.file_path, 'r') as path_file:
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
                line = Line(line_type, starting_point, endpoint, recipe)

                # If it's an EDGEDEL_LINE line type, set the thikness as well
                if line_type == 'EDGEDEL_LINE':
                    line.set_thikness(float(row[5].replace(',', '')))

                # Add the line to the list of lines
                self.lines.append(line)


    def optimize(self):
        """
        Groups Line objects, and runs a thread per optimization group.
        """

        # Group the lines
        groups = {}
        for line in self.lines:

            # Determine the line type and recipe number of the line
            line_type = line.get_line_type()
            recipe = line.get_recipe()

            # Determine to which group this line belongs
            if line_type == 'SCRIBE_LINE' and recipe == '2':
                group_name = 'SCRIBE_LINE2'
            elif line_type == 'REF':
                self.ref_line = line
                continue
            elif self.recipe_grouping:
                group_name = line_type
            else:
                group_name = line_type + recipe

            # Add the line to the group
            if group_name not in groups:
                groups[group_name] = [line]
            else:
                groups[group_name].append(line)

        # Probability of reproduction
        repro = 0.8

        # Probability of crossover
        crossover = 0.9

        # Probability of mutation
        mutation = 0.001

        # List containing all optimization processes
        processes = []

        # Manager which will take care of shared state (all the optimization
        # objects)
        process_manager = Manager()
        self.all_optimizations = process_manager.dict()

        # Start a thread for every group
        progress_bar_position = 0
        for group_name, group in groups.items():

            # Number of generations to evolve
            num_generations = 30*len(group)

            # Size of population per generation, determined by the size of the
            # optimization problem (number of nodes)
            pop_size = 10*len(group)

            # If there's not more that one line of this line type, don't
            # optimize it, because we will use it in the "final" optimization
            # run, when we optimize the optimizations together
            p = Process(target=self.start_process, args=(
                self.all_optimizations,
                group_name,
                group,
                pop_size,
                repro,
                crossover,
                mutation,
                num_generations,
                progress_bar_position
                ))
            p.name = group_name
            processes.append(p)
            p.start()
            progress_bar_position += 1

        # Wait for processes to finish before executing other code
        for process in processes:
            process.join()

        # Save the results in a list of tuples, containing the group name and
        # the node, but starting with REF
        # We have to respect the following order:
        # 1) REF
        # 2) SCRIBE_LINE (non 2 recipe)
        # 3) BUSBAR_LINE
        # 4) EDGEDEL_LINE
        # 5) SCRIBE_LINE2

        # The REF line is always first in the solution, if there is one
        if self.ref_line:
            self.result.append(self.ref_line)

        # Find all SCRIBE_LINE (non 2 recipe) groups
        for group_name, opt in self.all_optimizations.items():
            if 'SCRIBE_LINE' in group_name and '2' not in group_name:
                self.result.extend(
                        [groups[group_name][index] for index in
                            opt.best_result['solution']]
                        )

        # Find all BUSBAR_LINE groups
        for group_name, opt in self.all_optimizations.items():
            if 'BUSBAR_LINE' in group_name:
                self.result.extend(
                        [groups[group_name][index] for index in
                            opt.best_result['solution']]
                        )

        # Find all EDGEDEL_LINE groups
        for group_name, opt in self.all_optimizations.items():
            if 'EDGEDEL_LINE' in group_name:
                self.result.extend(
                        [groups[group_name][index] for index in
                            opt.best_result['solution']]
                        )

        # Find all SCRIBE_LINE2 groups
        for group_name, opt in self.all_optimizations.items():
            if 'SCRIBE_LINE2' == group_name:
                self.result.extend(
                        [groups[group_name][index] for index in
                            opt.best_result['solution']]
                        )

        ### Do the same thing for the initial result (used only for
        # visualization)
        if self.ref_line:
            self.initial.append(self.ref_line)

        # Find all SCRIBE_LINE (non 2 recipe) groups
        for group_name, opt in self.all_optimizations.items():
            if 'SCRIBE_LINE' in group_name and '2' not in group_name:
                self.initial.extend(
                        [groups[group_name][index] for index in
                            opt.initial_result['solution']]
                        )

        # Find all BUSBAR_LINE groups
        for group_name, opt in self.all_optimizations.items():
            if 'BUSBAR_LINE' in group_name:
                self.initial.extend(
                        [groups[group_name][index] for index in
                            opt.initial_result['solution']]
                        )

        # Find all EDGEDEL_LINE groups
        for group_name, opt in self.all_optimizations.items():
            if 'EDGEDEL_LINE' in group_name:
                self.initial.extend(
                        [groups[group_name][index] for index in
                            opt.initial_result['solution']]
                        )

        # Find all SCRIBE_LINE2 groups
        for group_name, opt in self.all_optimizations.items():
            if 'SCRIBE_LINE2' == group_name:
                self.initial.extend(
                        [groups[group_name][index] for index in
                            opt.initial_result['solution']]
                        )

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
            for line in self.result:
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
                formatted += "{:.3f}".format(line.get_thikness())
                formatted += ', '
                formatted += line.get_recipe()
                formatted += '\n'

                f.write(formatted)

    def start_process(self, all_optimizations, node_name, node, pop_size,
            repro, crossover, mutation, num_generations, progress_bar_position):
        """
        Method that creates the optimization object, starts the optimization
        and saves the result.

        Parameters
        ----------
        all_optimizations : dict
            Dictionary which will be used to store the optimization object that
            corresponds to every line group.
        node_name : str
            The name of the group which is being optimized.
        node : list of Line
            A list of Line objects, which represent the lines that belong to
            the group being optimized.
        pop_size : int
            Size of the population for every generation.
        repro : float
            Probability of reproduction.
        crossover : float
            Probability of crossover.
        mutation : float
            Probability of mutation.
        num_generations : int
            Number of iterations for which to run the algorithm.
        progress_bar_position : int
            Determines the row in which the progress bar will be displayed
            while optimizing.
        """

        # Initialize the optimization object
        opt = GeneticAlgorithm(node, pop_size, repro, crossover, mutation,
                num_generations, progress_bar_position)

        # Start the optimization
        opt.optimize()

        # Save the result (optimized object)
        all_optimizations[node_name] = opt

    def visualize(self):
        """
        Visualizes the result of the optimization, using the Visualizer class.
        """

        viz = Visualizer(self.result, self.initial)
        viz.visualize()


class GeneticAlgorithm():
    """
    Solves an instance of the travelling salesman problem, for the CNC machine.

    Finds the shortest cutting tool travel path (SCTTP) using the genetic
    algorithm optimization method.
    """

    def __init__(self, nodes, pop_size, repro, crossover, mutation,
            num_generations, progress_bar_position=0):
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

        # Get the nodes
        self.nodes = nodes

        # Number of genes per individual
        self.num_genes = len(self.nodes)

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

        # Progress bar position for tqdm
        self.progress_bar_position = progress_bar_position

    def generate_distance_matrix(self):
        """
        Generates matrix of Euclidian distances between every two nodes.
        """

        self.distance_matrix = np.zeros((self.num_genes, self.num_genes))
        for i in range(self.num_genes):
            for j in range(self.num_genes):
                self.distance_matrix[i, j] = np.linalg.norm(
                    self.nodes[i].get_endpoint() - self.nodes[j].get_starting_point()
                        )

    def evaluate_generation(self):
        """
        Evaluates the path cost and fitness of the whole generation.

        Path cost is calculated as the Euclidian distance between the second
        poin in a node and the first point in the next node. Fitness is
        calculated as the maxmimum possible path cost, minus the actual path
        cost.
        """

        self.path_cost = np.zeros(self.pop_size)

        # Maybe this is possible without doing a loop
        for individual in range(self.pop_size):
            for node in range(self.num_genes - 1):
                self.path_cost[individual] += self.distance_matrix[
                    self.population[individual][node],
                    self.population[individual][node + 1]
                        ]

        # Calculate the fitness
        self.fitness = self.num_genes*self.max_distance - self.path_cost

    def reproduction(self):
        """
        Determines which individuals get to move to the next generation (which
        ones get cloned).
        """

        # We're playing roulette, so we have to generate a ball that falls on
        # some individual
        for i in range(self.num_repro):

            # Generate the ball
            ball = np.ceil(np.random.uniform()*self.cumulative[-1]).astype(int)

            # Take the winner of the roulette game, and clone him into the next
            # generation
            index_of_winner = np.argmax(self.cumulative > ball).astype(int)
            self.population[i, :] = self.old_population[index_of_winner, :]

    def crossover(self):
        """
        Generates part of the population using crossover.

        Takes two individuals at a time, based on fitness and combines them,
        using the Order 1 Crossover method.
        """

        # We're playing roulette, so we have to generate a ball that falls on
        # some individual
        for i in range(self.num_cross//2):

            # Generate the ball
            ball = np.ceil(np.random.uniform()*self.cumulative[-1]).astype(int)

            # Take the winner of the roulette game
            index_of_winner = np.argmax(self.cumulative > ball).astype(int)
            parent1 = self.old_population[index_of_winner, :]

            # Do the whole thing again for the second parent
            ball = np.ceil(np.random.uniform()*self.cumulative[-1]).astype(int)
            index_of_winner = np.argmax(self.cumulative > ball).astype(int)
            parent2 = self.old_population[index_of_winner, :]

            # NOTE ovo sa verovatnocom ukrstanja je bas nekako jadno odradjeno,
            # vidi kako si uradio u algoritmu za VI
            # We're doing Odrder 1 crossover
            if np.random.uniform() < self.prob_cross:
                # Generate points, used for cutting out genetic material
                crp1 = np.random.randint(self.num_genes)
                crp2 = np.random.randint(crp1, self.num_genes)

                # Arrays to be populated with genetic material
                child1 = np.empty(self.num_genes)
                child2 = np.empty(self.num_genes)

                # Populate children with a cut of material
                child1[crp1:crp2] = parent1[crp1:crp2]
                child2[crp1:crp2] = parent2[crp1:crp2]

                # Fill in rest of material using Order 1 crossover
                other_genes_child1 = list(self.set_of_nodes - set(parent1[crp1:crp2]))
                other_genes_child2 = list(self.set_of_nodes - set(parent2[crp1:crp2]))
                for gen_num in range(self.num_genes - (crp2 - crp1)):
                    idx = (crp2 + gen_num) % self.num_genes
                    child1[idx] = other_genes_child1[gen_num]
                    child2[idx] = other_genes_child2[gen_num]

            else:
                child1 = parent1[:]
                child2 = parent2[:]

            # Add the children to the population
            self.population[self.num_repro + 2*i, :] = child1
            self.population[self.num_repro + 2*i + 1, :] = child2

    def mutation(self):
        """
        Mutates a set number of individuals in the population, by swapping two
        genes.
        """
        for i in range(self.num_mut):
            individual = int(np.ceil(np.random.uniform()*self.pop_size - 1))
            gene1 = int(np.ceil(np.random.uniform()*self.num_genes - 1))
            gene2 = int(np.ceil(np.random.uniform()*self.num_genes - 1))
            gene = self.population[individual, gene1]
            self.population[individual, gene1] = self.population[
                    individual, gene2
                    ]
            self.population[individual, gene2] = gene

    def optimize(self):
        """
        Runs the optimization algorithm trying to find the shortest path.
        """

        # Numbers representing the nodes
        nodes = np.arange(self.num_genes)

        # Generate the initial population
        self.population = np.ndarray((0, self.num_genes)).astype(int)
        for i in range(self.pop_size):
            self.population = np.vstack((
                self.population,
                np.random.permutation(nodes).reshape(1, self.num_genes),
                ))

        # Get the name of the process
        process_name = current_process().name

        # Number of iterations of one run
        for generation in tqdm(
                range(self.num_generations),
                position=self.progress_bar_position,
                desc=process_name
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
                        ]
                self.best_result['path_cost'] = self.path_cost.min()

                # If this is the 1st iteration, save the result for comparison
                if generation < 1:
                    self.initial_result['solution'] = self.population[0]
                    self.initial_result['path_cost'] = self.path_cost[0]

            # Calculate cumulative sum for roulette game (used for
            # reproduction and crossover)
            self.cumulative = self.fitness.cumsum()

            # Generate a placeholder for the new population, and remember
            # the old population
            self.old_population = self.population[:]
            self.population = np.zeros(
                    (
                        self.pop_size,
                        self.num_genes
                        )
                    ).astype(int)


            # Perform reproduction
            self.reproduction()

            # Perform crossover
            self.crossover()

            # Perform mutation
            self.mutation()


class Line():
    """
    Line which represents where the CNC head will perform cutting.
    """

    def __init__(self, line_type, starting_point, endpoint, recipe):
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

        return self.starting_point

    def get_endpoint(self):
        """
        Returns the endpoint of a line.

        Returns
        -------
        endpoint : np.array
            Numpy array of two coordinates, X2 and Y2, representing the
            endpoint of cutting.
        """
        return self.endpoint

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
