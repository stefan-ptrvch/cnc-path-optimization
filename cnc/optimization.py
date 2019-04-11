import csv
import numpy as np


class ShortestPath():
    """
    Finds the shortest cutting tool travel path (SCTTP) using the genetic
    algorithm.
    """

    def __init__(self, file_path, pop_size, repro, crossover, mutation,
            num_generations, num_runs):
        """
        Initializes the optimization object with all the needed values to run
        the optimization.
        """

        # Path to file containing code of CNC path
        self.file_path = file_path

        # Get the nodes
        self.generate_nodes_from_file()

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

        # Number of runs of the algorithm
        self.num_runs = num_runs

        # Best result overall
        self.best_result = {'solution': [], 'path_cost': np.inf}

        # Best result per generation per run
        self.best_result_per_gen = {'solution': [], 'path_cost': np.inf}

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

        self.nodes = np.ndarray((0, 2, 2))
        with open(self.file_path, 'r') as path_file:
            reader = csv.reader(path_file, delimiter=' ')
            for row in reader:
                if 'SCRIBE_LINE' == row[0]:
                    node = np.array([
                        float(row[1].replace(',', '')),
                        float(row[2].replace(',', '')),
                        float(row[3].replace(',', '')),
                        float(row[4].replace(',', '')),
                        ]).reshape((1, 2, 2))
                    self.nodes = np.vstack((self.nodes, node))


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
        Runs the optimizational algorithm num_runs times, trying to find the
        shortest path.
        """

        # Numbers representing the nodes
        nodes = np.arange(self.num_genes)

        # Number of runts to start the optimization from scratch
        for run in range(self.num_runs):

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
