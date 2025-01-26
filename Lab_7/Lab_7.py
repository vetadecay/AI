import numpy as np

def func_y(x):
    return np.sin(np.abs(x)) * np.cos(3 * x / 2)

def func_z(x, y):
    return x * np.sin(x + y)


class Individual:
    def __init__(self, genes, fitness):
        self.genes = genes
        self.fitness = fitness

def initialize_population(population_size, num_genes, gene_min, gene_max):
    return [
        Individual(genes=np.random.uniform(gene_min, gene_max, num_genes), fitness=None)
        for _ in range(population_size)
    ]

def evaluate_fitness(population, func):
    for individual in population:
        individual.fitness = func(*individual.genes)

def select_parents(population):
    population.sort(key=lambda ind: ind.fitness)
    return population[:2]

def crossover(parent1, parent2):
    child_genes = (parent1.genes + parent2.genes) / 2
    return Individual(genes=child_genes, fitness=None)

def mutate(individual, mutation_rate, gene_min, gene_max):
    if np.random.rand() < mutation_rate:
        mutation_index = np.random.randint(len(individual.genes))
        individual.genes[mutation_index] = np.random.uniform(gene_min, gene_max)

def genetic_algorithm(
    population_size, num_genes, gene_min, gene_max, mutation_rate, generations, func, min_or_max
):
    population = initialize_population(population_size, num_genes, gene_min, gene_max)

    for generation in range(generations):
        evaluate_fitness(population, func)
        if min_or_max == "min":
            population.sort(key=lambda ind: ind.fitness)
        elif min_or_max == "max":
            population.sort(key=lambda ind: ind.fitness, reverse=True)

        parents = select_parents(population)
        new_population = []

        for _ in range(population_size):
            child = crossover(parents[0], parents[1])
            mutate(child, mutation_rate, gene_min, gene_max)
            new_population.append(child)

        population = new_population

    evaluate_fitness(population, func)
    best_individual = (
        min(population, key=lambda ind: ind.fitness)
        if min_or_max == "min"
        else max(population, key=lambda ind: ind.fitness)
    )

    return best_individual


best_individual_y = genetic_algorithm(
    population_size=50,
    num_genes=1,
    gene_min=-5.0,
    gene_max=5.0,
    mutation_rate=0.35,
    generations=100,
    func=lambda x: func_y(x),
    min_or_max="min"
)

print("Minimum of func_y:", best_individual_y.genes, "Value:", best_individual_y.fitness)


best_individual_z = genetic_algorithm(
    population_size=150,
    num_genes=2,
    gene_min=-5.0,
    gene_max=5.0,
    mutation_rate=0.35,
    generations=500,
    func=lambda x, y: func_z(x, y),
    min_or_max="max"
)

print("Maximum of func_z:", best_individual_z.genes, "Value:", best_individual_z.fitness)
