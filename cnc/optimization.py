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
    Parses the input file, and generates a bunch of ShortestPath optimizers,
    for different line types and for recipe groupings, if it's required. It
    then runs the optimizers in parallel and saves the result.
    """

    def __init__(self, file_path, recipe_grouping=True):
        """
        Initializes the object.
        """

        # Path to file which contains line coordinates which need to be
        # optimized
        self.file_path = file_path

        # Whether to do recipe grouping
        self.recipe_grouping = recipe_grouping

        # Generate dictionary of nodes
        self.generate_nodes_from_file()


    def generate_nodes_from_file(self):
        """
        Parses input file and generates the list of nodes that need to be
        optimized.

        Nodes are represented by 3D arrays of size (1, 2, 2), where the 2nd
        dimension is the start/end poin and the 3rd is the x/y coordinate of
        the point.

        The list of nodes is represented by a 3D array of size (N, 2, 2) where
        N represents the total number of nodes (total number of lines in file).
        """

        # Dictionary which will contain nodes, grouped in a specific order
        self.nodes = {}

        with open(self.file_path, 'r') as path_file:
            reader = csv.reader(path_file, delimiter=' ')
            for row in reader:

                # The name of the line
                line_type = row[0]

                # The number of the recipe
                recipe = row[5]

                # Determine the name of the group
                # We make groups of nodes in the following manner, if we're
                # using grouping
                #  1) REF
                #  2) SCRIBE_LINE --> Except for Recipe #2
                #  3) BUSBAR_LINE --> All Recipes
                #  4) EDGEDEL_LINE --> All Recipes
                #  5) SCRIBE_LINE --> Only Recipe #2
                # If we're not using grouping, we split the line types by
                # group, but keep the line order intact, and SCRIBE_LINE recipe
                # number 2 at the end
                if line_type == 'SCRIBE_LINE' and recipe == '2':
                    group_name = 'SCRIBE_LINE2'
                elif line_type == 'REF':
                    group_name = 'REF'
                elif self.recipe_grouping:
                    group_name = line_type
                else:
                    group_name = line_type + recipe

                # The node itself
                # The coordinates in the file are Y1, X1, Y2, X2, and we
                # want to store them as X1, Y1, X2, Y2
                # We skip the 1st entry since it contains the type of line
                node = np.array([
                    float(row[2].replace(',', '')),
                    float(row[1].replace(',', '')),
                    float(row[4].replace(',', '')),
                    float(row[3].replace(',', '')),
                    ]).reshape((1, 2, 2))

                # Add line to dictionary, if it isn't in there
                if group_name not in self.nodes:
                    self.nodes[group_name] = node

                else:
                    # Add the node to the existing nodes
                    self.nodes[group_name] = np.vstack((
                        self.nodes[group_name],
                        node
                        ))


    def optimize(self):
        """
        Runs optimizations for all the line types in parallel.
        """

        # Size of population per generation
        pop_size = 300

        # Probability of reproduction
        repro = 0.8

        # Probability of crossover
        crossover = 0.9

        # Probability of mutation
        mutation = 0.001

        # Number of generations to evolve
        num_generations = 1000

        # List containing all optimization processes
        processes = []

        # Manager which will take care of shared state (all the optimization
        # objects)
        process_manager = Manager()
        self.all_optimizations = process_manager.dict()

        # For every line type, and recipe number (if we use recipe grouping),
        # start an optimization
        progress_bar_position = 0
        for group_name, group in self.nodes.items():

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

        self.result = [('REF', self.nodes['REF'][0])]

        # Find all SCRIBE_LINE (non 2 recipe) groups
        for group_name, opt in self.all_optimizations.items():
            if 'SCRIBE_LINE' in group_name and '2' not in group_name:
                for node_number in opt.best_result['solution']:
                    self.result.append((
                        group_name,
                        self.nodes[group_name][node_number]))

        # Find all BUSBAR_LINE groups
        for group_name, opt in self.all_optimizations.items():
            if 'BUSBAR_LINE' in group_name:
                for node_number in opt.best_result['solution']:
                    self.result.append((
                        group_name,
                        self.nodes[group_name][node_number]))

        # Find all EDGEDEL_LINE groups
        for group_name, opt in self.all_optimizations.items():
            if 'EDGEDEL_LINE' in group_name:
                for node_number in opt.best_result['solution']:
                    self.result.append((
                        group_name,
                        self.nodes[group_name][node_number]))

        # Find all SCRIBE_LINE2 groups
        for group_name, opt in self.all_optimizations.items():
            if 'SCRIBE_LINE2' == group_name:
                for node_number in opt.best_result['solution']:
                    self.result.append((
                        group_name,
                        self.nodes[group_name][node_number]))

        ### Do the same thing for the initial result (used only for
        # visualization)
        self.initial = [('REF', self.nodes['REF'][0])]

        # Find all SCRIBE_LINE (non 2 recipe) groups
        for group_name, opt in self.all_optimizations.items():
            if 'SCRIBE_LINE' in group_name and '2' not in group_name:
                for node_number in opt.initial_result['solution']:
                    self.initial.append((
                        group_name,
                        self.nodes[group_name][node_number]))

        # Find all BUSBAR_LINE groups
        for group_name, opt in self.all_optimizations.items():
            if 'BUSBAR_LINE' in group_name:
                for node_number in opt.initial_result['solution']:
                    self.initial.append((
                        group_name,
                        self.nodes[group_name][node_number]))

        # Find all EDGEDEL_LINE groups
        for group_name, opt in self.all_optimizations.items():
            if 'EDGEDEL_LINE' in group_name:
                for node_number in opt.initial_result['solution']:
                    self.initial.append((
                        group_name,
                        self.nodes[group_name][node_number]))

        # Find all SCRIBE_LINE2 groups
        for group_name, opt in self.all_optimizations.items():
            if 'SCRIBE_LINE2' == group_name:
                for node_number in opt.initial_result['solution']:
                    self.initial.append((
                        group_name,
                        self.nodes[group_name][node_number]))


    def save(self, file_name):
        """
        Saves the results of the optimization to a file.
        """
        pass


    def start_process(self, all_optimizations, node_name, node, pop_size,
            repro, crossover, mutation, num_generations, progress_bar_position):
        """
        Method that creates initializes the optimization object, starts the
        optimization and saves the result.
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
        Visualizes the result of the optimization.
        """

        viz = Visualizer(self.result, self.initial)
        viz.visualize_solution()


class GeneticAlgorithm():
    """
    Finds the shortest cutting tool travel path (SCTTP) using the genetic
    algorithm.
    """

    def __init__(self, nodes, pop_size, repro, crossover, mutation,
            num_generations, progress_bar_position=0):
        """
        Initializes the optimization object with all the needed values to run
        the optimization.
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

        # We need an even number of crossings (because we need pairs)
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
        # optimizational puproses)
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
                    self.nodes[i][1] - self.nodes[j][0]
                        )


    def evaluate_generation(self):
        """
        Evaluates the path cost and fitness of the whole generation.

        Path cost is calculated as the Euclidian distance between the second
        poin in a node and the first point in the next node.

        Fitness is calculated as the maxmimum path cost of the generation,
        minus the path cost.
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


    def crossing(self):
        """
        Takes individuals and crosses them to generate new individuals for the
        population.
        """
        # NOTE videti da li moze sve ovo astype da se obrise (to bi trebalo da
        # ubrza algoritam)

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
            if np.random.uniform() < self.prob_cross:
                crossover_point = int(np.ceil(
                    np.random.uniform()*(self.num_genes - 1)
                    ))

                child1 = np.hstack(
                        (parent1[:crossover_point], parent2[crossover_point:])
                        )
                child2 = np.hstack(
                        (parent2[:crossover_point], parent1[crossover_point:])
                        )

                # Now we have to fix the children, so as to not have repeating
                # nodes
                # NOTE ovo verovatno moze brze da se implementira a verovatno i
                # postoji bolji nacin za ispravljanje dece
                missing_nodes = self.set_of_nodes - set(child1)
                for node in missing_nodes:
                    for j in range(child1.size):
                        if child1[j] in child1[j + 1:]:
                            child1[j] = node

                missing_nodes = self.set_of_nodes - set(child2)
                for node in missing_nodes:
                    for j in range(child2.size):
                        if child2[j] in child2[j + 1:]:
                            child2[j] = node
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
        Runs the optimizational algorithm trying to find the shortest path.
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
            # reproduction and crossing)
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

            # Perform crossing
            self.crossing()

            # Perform mutation
            self.mutation()
