import csv
import numpy as np
from multiprocessing import Process, Manager
from cnc.visualization import Visualizer


class CNCOptimizer():
    """
    Parses the input file, and generates a bunch of ShortestPath optimizers,
    for different line types and for recipe groupings, if it's required. It
    then runs the optimizers in parallel and saves the result.
    """

    def __init__(self, file_path, recipe_grouping=False):
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

        # Dictionary which groups nodes based on line type, or line type/recipe
        # type
        self.nodes = {}

        with open(self.file_path, 'r') as path_file:
            reader = csv.reader(path_file, delimiter=' ')
            for row in reader:

                # Determine the group name
                if self.recipe_grouping:
                    group_name = row[0] + row[5]
                else:
                    group_name = row[0]

                # Add group to dictionary, if it isn't in there
                if group_name not in self.nodes:
                    self.nodes[group_name] = np.ndarray((0, 2, 2))

                # The coordinates in the file are Y1, X1, Y2, X2, and we
                # want to store them as X1, Y1, X2, Y2
                # We skip the 1st entry since it contains the type of line
                node = np.array([
                    float(row[2].replace(',', '')),
                    float(row[1].replace(',', '')),
                    float(row[4].replace(',', '')),
                    float(row[3].replace(',', '')),
                    ]).reshape((1, 2, 2))

                # Add the node to the existing nodes
                self.nodes[group_name] = np.vstack(
                        (self.nodes[group_name], node)
                        )


    def optimize(self):
        """
        Runs optimizations for all the line types in parallel.
        """

        # Size of population per generation
        pop_size = 10

        # Probability of reproduction
        repro = 0.8

        # Probability of crossover
        crossover = 0.9

        # Probability of mutation
        mutation = 0.001

        # Number of generations to evolve
        num_generations = 10

        # Whether to remember best population, for debugging/visualization
        debug = False

        # List containing all optimization processes
        processes = []

        # Manager which will take care of shared state (all the optimization
        # objects)
        process_manager = Manager()
        self.all_optimizations = process_manager.dict()

        # For every line type, and recipe number (if we use recipe grouping),
        # start an optimization
        for group_name, group in self.nodes.items():

            # If there's not more that one line of this line type, don't
            # optimize it, because we will use it in the "final" optimization
            # run, when we optimize the optimizations together
            if group.shape[0] > 1:
                p = Process(target=self.start_process, args=(
                    self.all_optimizations,
                    group_name,
                    group,
                    pop_size,
                    repro,
                    crossover,
                    mutation,
                    num_generations,
                    debug
                    ))
                processes.append(p)
                p.start()

        # Wait for processes to finish before executing other code
        for process in processes:
            process.join()

        # Run the last optimization, which optimizes the best paths of all
        # subgroups
        # These "group nodes" will be the starting point of the first line of
        # the best path, and the ending point of the last line of the best
        # path, so that we practically treat the whole "best path" of an
        # optimization as one line, and then we optimize all the "best paths"
        # of all groups together, as a set of line, since it does not matter
        # how the tool end up where it is (we expect that it did the least
        # ammount of travelling possible)

        # This is the empty group nodes array
        #  self.group_nodes = np.ndarray((0, 2, 2))

        #  # This is a group-to-node mapping dictionary, which we need to later
        #  # sort the groups correctly, since the optimization algorithm will give
        #  # us the correct order of the nodes
        #  group_to_node = {}
        #  i = 0

        #  # Go through all the results, and extract the start and finish of the
        #  # optimization path
        #  for opt_name, opt in self.all_optimizations.items():

            #  # Get the best path
            #  solution = opt.best_result['solution']

            #  # Extract the beginning and the end of the best path and treat it
            #  # as one node (one line)
            #  # We use the starting position of the first node in the solution
            #  # path, and the end position of the last node in the solution path
            #  node = np.array([
                #  opt.nodes[solution[0]][0],
                #  opt.nodes[solution[-1]][1]
                #  ]).reshape((1, 2, 2))

            #  # Now add it to the list of nodes
            #  self.group_nodes = np.vstack((self.group_nodes, node))
            #  group_to_node[opt_name] = i
            #  i += 1

        #  # Now, add all the lines types/recipe groupings of which there are only
        #  # one (which could not be optimized)
        #  for group_name, group in self.nodes.items():
            #  if group.shape[0] == 1:
                #  self.group_nodes = np.vstack((
                    #  self.group_nodes,
                    #  group
                    #  ))
                #  group_to_node[group_name] = i
                #  print(group_name)
                #  i += 1
                #  print(group_to_node)

        #  self.opt = GeneticAlgorithm(self.group_nodes, pop_size, repro, crossover,
                    #  mutation, num_generations, debug)

        #  # Start the optimization
        #  self.opt.optimize()

        #  # Write the output
        #  # Order the groups, based on the optimization
        #  group_order = [
                #  self.nodes.keys()[index] for index in
                #  self.opt.best_result['solution']
                #  ]
        #  print(group_order)

        #  # Write solution to file
        #  with open('optimized.code', mode='w') as csv_file:
            #  writer = csv.writer(csv_file, delimiter=',')
            #  for group in group_order:
                #  # Coordinates are saved as X1, Y1, X2, Y2, but we need to save
                #  # them as Y1, X1, Y2, X2
                #  if type(self.all_optimizations[group]) is np.ndarray:
                    #  row = []
                    #  row.append(group)
                    #  row.append(self.all_optimizations[group][0][0, 1])
                    #  row.append(self.all_optimizations[group][0][0, 0])
                    #  row.append(self.all_optimizations[group][0][1, 1])
                    #  row.append(self.all_optimizations[group][0][1, 0])
                    #  writer.writerow(row)
                #  else:
                    #  path = self.all_optimizations[group].best_result['solution']
                    #  nodes = self.all_optimizations[group].nodes
                    #  for node in path:
                        #  row = []
                        #  row.append(group)
                        #  row.append(nodes[node][0, 1])
                        #  row.append(nodes[node][0, 0])
                        #  row.append(nodes[node][1, 1])
                        #  row.append(nodes[node][1, 0])
                        #  writer.writerow(row)


    def save(self, file_name):
        """
        Saves the results of the optimization to a file.
        """
        pass


    def start_process(self, all_optimizations, node_name, node, pop_size,
            repro, crossover, mutation, num_generations, debug):
        """
        Method that creates initializes the optimization object, starts the
        optimization and saves the result.
        """

        # Initialize the optimization object
        opt = GeneticAlgorithm(node, pop_size, repro, crossover, mutation,
                num_generations, debug)

        # Start the optimization
        opt.optimize()

        # Save the result (optimized object)
        all_optimizations[node_name] = opt


    def visualize(self):
        """
        Visualizes the result of the optimization.
        """
        return
        viz = Visualizer(opt)
        #  viz.visualize_solution()


class GeneticAlgorithm():
    """
    Finds the shortest cutting tool travel path (SCTTP) using the genetic
    algorithm.
    """

    def __init__(self, nodes, pop_size, repro, crossover, mutation,
            num_generations, debug=False):
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

        # Switch for saving data if we need it for debugging/visualization
        self.debug = debug

        # All of the populations of the optimization, used for
        # debugging/visualization
        self.pop_history = np.ndarray((0, self.num_genes))

        # Path cost of generation
        self.path_cost = None

        # Fitness of generation, which is inverse normalized path cost (for
        # optimizational puproses)
        self.fitness = None


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

        # Number of iterations of one run
        for generation in range(self.num_generations):

            # Evaluate the current generation
            self.evaluate_generation()

            ## DEBUG
            #  print(generation, self.path_cost.mean())
            ## DEBUG

            # If we are debuggin/visualizing, remember the run
            if self.debug:
                self.pop_history = np.vstack((self.pop_history, self.population))

            # Get the best individual and his path cost
            if self.best_result['path_cost'] > self.path_cost.min():
                self.best_result['solution'] = self.population[
                        self.path_cost.argmin()
                        ]
                self.best_result['path_cost'] = self.path_cost.min()

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
