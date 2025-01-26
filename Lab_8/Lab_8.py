import random
import numpy as np


class Individual:
    def __init__(self, genes, fitness=None):
        self.genes = genes
        self.fitness = fitness


def initialize_population(population_size, nodes_num, switches_num):
    population = []
    for _ in range(population_size):
        genes = np.random.randint(1, switches_num + 1, size=nodes_num)
        population.append(Individual(genes))
    return population


def evaluate_fitness(individual, nodes_num, switches_num):
    switch_load = [np.sum(individual.genes == i) for i in range(1, switches_num + 1)]
    fitness = -np.var(switch_load)
    individual.fitness = fitness


def select_parents(population):
    sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    return sorted_population[:2]


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1.genes) - 1)
    child_genes = np.concatenate((parent1.genes[:crossover_point], parent2.genes[crossover_point:]))
    return Individual(child_genes)


def mutate(individual, mutation_rate, switches_num):
    if random.random() < mutation_rate:
        mutation_index = random.randint(0, len(individual.genes) - 1)
        individual.genes[mutation_index] = random.randint(1, switches_num)


def genetic_algorithm(nodes_num, switches_num, population_size, generations, mutation_rate):
    population = initialize_population(population_size, nodes_num, switches_num)

    for generation in range(generations):
        for individual in population:
            evaluate_fitness(individual, nodes_num, switches_num)


        parents = select_parents(population)


        new_population = []
        for _ in range(population_size):
            child = crossover(parents[0], parents[1])
            mutate(child, mutation_rate, switches_num)
            new_population.append(child)

        population = new_population


    for individual in population:
        evaluate_fitness(individual, nodes_num, switches_num)

    best_individual = max(population, key=lambda ind: ind.fitness)
    return best_individual


if __name__ == '__main__':
    nodes_num = 15
    switches_num = 4

    population_size = 200
    generations = 50
    mutation_rate = 0.2

    best_individual = genetic_algorithm(nodes_num, switches_num, population_size, generations, mutation_rate)

    print(f"Switches: {best_individual.genes}")
    print(f"Node number: {[node for node in range(1, nodes_num + 1)]}")
