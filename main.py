"""Test run: python -m mutational_robustness.main"""
import numpy as np

from mutational_robustness.fitness_landscape import simple_landscape
from mutational_robustness.fitness_landscape import adjacent_landscape
from mutational_robustness.fitness_landscape import hybrid_landscape
from mutational_robustness.wright_fisher import simple_landscape_wright_fisher
from mutational_robustness.wright_fisher import adjacent_landscape_wright_fisher
from mutational_robustness.wright_fisher import hybrid_landscape_wright_fisher

def _logger(
    counts: 'numpy.array, each element corresponds to number of individuals',
    fitness: 'numpy.array, each element corresponds to fitness of each bin',
    iteration: 'int, current iteration',
    counts_logger: 'numpy.array, each row = [mean, variance] counts for each iteration',
    fitness_logger: 'numpy.array, each row = [mean, variance] fitness for each iteration',
    raw_logger: 'list, at each iteration append counts to it',
    start: 'int, original count for the first element, default to 0' = 0
):
    """Helper function for logging counts and fitness statistics."""
    if np.sum(counts) > 0:
        counts_pop = np.repeat(np.arange(start, start + fitness.shape[0]), counts)
        fitness_pop = np.repeat(fitness, counts)
        counts_logger[iteration, :] = np.mean(counts_pop), np.var(counts_pop)
        fitness_logger[iteration, :] = np.mean(fitness_pop), np.var(fitness_pop)
    else:
        counts_logger[iteration, :] = 0.0, 0.0
        fitness_logger[iteration, :] = 0.0, 0.0
    if raw_logger:
        raw_logger.append(counts)

def simulate_simple_landscape(
    s: 'float, mutational effect',
    eps: 'float, epistasis',
    l: 'int, landscape size',
    n: 'int, population size',
    u_ben: 'float, beneficial mutation rate',
    u_del: 'float, deleterious mutation rate',
    generations: 'int, max number of iterations'
):
    counts_logger = np.zeros((generations, 2))
    fitness_logger = np.zeros((generations, 2))
    fitness = simple_landscape(s, eps, k=np.arange(l + 1))
    counts = np.zeros(l + 1, dtype=int)
    counts[0] = n
    for iteration in range(generations):
        counts = simple_landscape_wright_fisher(
            n, l, u_ben, u_del, counts, fitness,
        )
        _logger(
            counts, fitness, iteration, counts_logger, fitness_logger, None,
        )
    return counts_logger, fitness_logger

def simulate_adjacent_landscape(
    s_l: 'float, mutational effect on the left fitness landscape',
    s_r: 'float, mutational effect on the right fitness landscape',
    eps_l: 'float, epistasis of the left fitness landscape',
    eps_r: 'float, epistasis of the right fitness landscape',
    l_l: 'int, size of the left fitness landscape',
    l_r: 'int, size of the right fitness landscape',
    n: 'int, population size',
    u_ben: 'float, beneficial mutation rate',
    u_del: 'float, deleterious mutation rate',
    generations: 'int, max number of iterations'
):
    counts_logger = np.zeros((generations, 2))
    fitness_logger = np.zeros((generations, 2))
    # Log fitness and counts for each part of the landscape separately.
    counts_logger_l = np.zeros((generations, 2))
    fitness_logger_l = np.zeros((generations, 2))
    counts_logger_r = np.zeros((generations, 2))
    fitness_logger_r = np.zeros((generations, 2))
    fitness = adjacent_landscape(
        s_l, s_r, eps_l, eps_r, l_l, l_r, k=np.arange(l_l + l_r + 2)
    )
    counts = np.zeros(l_l + l_r + 2, dtype=int)
    counts[np.random.choice(l_l + l_r + 2)] = n
    raw_logger = [counts, ]
    for iteration in range(generations):
        counts = adjacent_landscape_wright_fisher(
            n, l_l, l_r, u_ben, u_del, counts, fitness,
        )
        _logger(
            counts, fitness, iteration, counts_logger, fitness_logger, raw_logger,
        )
        _logger(
            counts[:(l_l + 1)], fitness[:(l_l + 1)], iteration, counts_logger_l, fitness_logger_l, None,
        )
        _logger(
            counts[(l_l + 1):], fitness[(l_l + 1):], iteration, counts_logger_r, fitness_logger_r, None,
            start=l_l + 1,
        )
    return (
        counts_logger,
        fitness_logger,
        counts_logger_l,
        fitness_logger_l,
        counts_logger_r,
        fitness_logger_r,
        raw_logger
    )

def simulate_hybrid_landscape(
    s_l: 'float, mutational effect on the left fitness landscape',
    s_r: 'float, mutational effect on the right fitness landscape',
    eps_l: 'float, epistasis of the left fitness landscape',
    eps_r: 'float, epistasis of the right fitness landscape',
    l_l: 'int, size of the left fitness landscape',
    l_r: 'int, size of the right fitness landscape',
    n: 'int, population size',
    u_ben: 'float, beneficial mutation rate',
    u_del: 'float, deleterious mutation rate',
    p_r: 'float, probability to take a path to the right landscape',
    generations: 'int, max number of iterations'
):
    counts_logger = np.zeros((generations, 2))
    fitness_logger = np.zeros((generations, 2))
    # Log fitness and counts for each part of the landscape separately.
    counts_logger_l = np.zeros((generations, 2))
    fitness_logger_l = np.zeros((generations, 2))
    counts_logger_r = np.zeros((generations, 2))
    fitness_logger_r = np.zeros((generations, 2))
    fitness = hybrid_landscape(
        s_l, s_r, eps_l, eps_r, l_l, l_r, k=np.arange(l_l + l_r + 1)
    )
    counts = np.zeros(l_l + l_r + 1)
    counts[l_l] = n
    raw_logger = [counts, ]
    for iteration in range(generations):
        counts = hybrid_landscape_wright_fisher(
            n, l_l, l_r, u_ben, u_del, p_r, counts, fitness,
        )
        _logger(
            counts, fitness, iteration, counts_logger, fitness_logger, raw_logger,
        )
        # Notice that the peak is "shared" between the two landscapes.
        _logger(
            counts[:l_l], fitness[:l_l], iteration, counts_logger_l, fitness_logger_l, None,
        )
        _logger(
            counts[(l_l + 1):], fitness[(l_l + 1):], iteration, counts_logger_r, fitness_logger_r, None,
            start=l_l + 1,
        )
    return (
        counts_logger,
        fitness_logger,
        counts_logger_l,
        fitness_logger_l,
        counts_logger_r,
        fitness_logger_r,
        raw_logger
    )

if __name__ == '__main__':
    # for testing purposes only -- for actual parameters see the manuscript.
    s = 0.1
    eps = 0
    eps_l = 0.1
    eps_r = -0.1
    l = 5
    n = 100
    u_ben = 0.0
    u_del = 0.1
    p_r = 0.5
    generations = 10
    # counts_logger, fitness_logger = simulate_simple_landscape(
    #     s, eps, l, n, u_ben, u_del, generations,
    # )
    # (
    #     counts_logger,
    #     fitness_logger,
    #     counts_logger_l,
    #     fitness_logger_l,
    #     counts_logger_r,
    #     fitness_logger_r,
    #     raw_logger
    # ) = simulate_adjacent_landscape(
    #     s, s, eps_l, eps_r, l, l, n, u_ben, u_del, generations,
    # )
    (
        counts_logger,
        fitness_logger,
        counts_logger_l,
        fitness_logger_l,
        counts_logger_r,
        fitness_logger_r,
        raw_logger
    ) = simulate_hybrid_landscape(
        s, s, eps_l, eps_r, l, l, n, u_ben, u_del, p_r, generations,
    )
    print(counts_logger)
    print(fitness_logger)
    print(counts_logger_l)
    print(counts_logger_r)
    print(raw_logger)
