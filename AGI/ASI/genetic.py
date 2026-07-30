"""Genetic Algorithm implementation.

The Chromosome trait defines the interface: mutate, crossover, fitness.
For selection, this provides RouletteWheel and Tournament strategies.
"""

import random
import math

class Chromosome:
    def mutate(self, rng):
        raise NotImplementedError

    def crossover(self, other, rng):
        raise NotImplementedError

    def fitness(self):
        raise NotImplementedError

def roulette_wheel_select(population, rng):
    sum_fitness = 0.0
    zero_fit = []
    for c in population:
        f = c.fitness()
        if f == 0.0:
            zero_fit.append(c)
        else:
            sum_fitness += 1.0 / f
    if len(zero_fit) >= 2:
        return zero_fit[0], zero_fit[1]

    parents = []
    for _ in range(2):
        spin = rng.random() * sum_fitness
        for c in population:
            f = c.fitness()
            if f == 0.0:
                continue
            w = 1.0 / f
            if spin <= w:
                parents.append(c)
                break
            spin -= w
    return parents[0], parents[1]

def tournament_select(population, rng, k=3):
    if k < 2:
        raise ValueError("k must be >= 2")
    picked = set()
    while len(picked) < min(k, len(population)):
        picked.add(rng.randint(0, len(population) - 1))
    sorted_idxs = sorted(picked, key=lambda i: population[i].fitness())
    return population[sorted_idxs[0]], population[sorted_idxs[1]]

class GeneticAlgorithm:
    def __init__(self, population, threshold, max_generations=100,
                 mutation_chance=0.2, crossover_chance=0.4, rng=None):
        self.rng = rng or random
        self.population = population
        self.threshold = threshold
        self.max_generations = max_generations
        self.mutation_chance = mutation_chance
        self.crossover_chance = crossover_chance

    def solve(self):
        generations = 1
        while generations <= self.max_generations:
            self.population.sort(key=lambda c: c.fitness())

            if self.population[0].fitness() <= self.threshold:
                return self.population[0]

            for c in self.population:
                if self.rng.random() <= self.mutation_chance:
                    c.mutate(self.rng)

            new_pop = []
            while len(new_pop) < len(self.population):
                p1, p2 = roulette_wheel_select(self.population, self.rng)
                if self.rng.random() <= self.crossover_chance:
                    child = p1.crossover(p2, self.rng)
                    new_pop.append(child)
                else:
                    new_pop.append(p1)
                    new_pop.append(p2)

            while len(new_pop) > len(self.population):
                new_pop.pop()
            self.population = new_pop
            generations += 1

        return None


if __name__ == "__main__":
    class StringChromosome(Chromosome):
        def __init__(self, secret, chars):
            self.secret = secret
            self.chars = chars
            self.genes = [random.choice(chars) for _ in range(len(secret))]

        def mutate(self, rng):
            idx = rng.randint(0, len(self.secret) - 1)
            self.genes[idx] = rng.choice(self.chars)

        def crossover(self, other, rng):
            child = StringChromosome.__new__(StringChromosome)
            child.secret = self.secret
            child.chars = self.chars
            child.genes = []
            for i in range(len(self.secret)):
                child.genes.append(self.genes[i] if rng.random() < 0.5 else other.genes[i])
            return child

        def fitness(self):
            return sum(1 for i in range(len(self.secret))
                       if self.genes[i] != self.secret[i])

    secret = "hi"
    chars = [chr(c) for c in range(ord('a'), ord('z') + 1)]
    pop = [StringChromosome(secret, chars) for _ in range(50)]
    ga = GeneticAlgorithm(pop, 0, max_generations=500, mutation_chance=0.2, crossover_chance=0.4)
    result = ga.solve()
    assert result is not None
    assert "".join(result.genes) == secret
    print("genetic: OK")
