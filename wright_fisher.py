"""
Simulate the Wright-Fisher process.

Write-Fisher process:
    Notations:
        N: population size
        u_ben: beneficial mutation rate
        u_del: deleterious mutation rate
    
    Description:
        Each iteration of the process has three steps:
            1. generate N "newborns" from the previous iteration's population,
            at random, with replacement, proportional to each individual's
            fitness
            2. simulate "mutations" for each newborn, where the number of
            deleterious (beneficial) mutations is assumed to be following a
            Poisson process with lambda = u_del (u_ben)
            3. the details of the third step are dependent on the underlying
            fitness landscape (see below).
"""
import numpy as np

def _generate_newborn_and_mutation(
    n: 'int, population size',
    l: 'int, landscape size',
    u_ben: 'float, beneficial mutation rate',
    u_del: 'float, deleterious mutation rate',
    counts: 'numpy.array, each element corresponds to number of individuals',
    fitness: 'numpy.array, each element corresponds to fitness of each bin',
) -> 'tuple of numpy.array that correspond to (newborn, mut_ben, mut_del)':
    """Helper function for the Wright Fisher process."""
    # Calculate the fitness mass in each bin.
    fitness_mass = counts * fitness
    fitness_prob = fitness_mass / np.sum(fitness_mass)
    # Generate newborns.
    newborn = np.random.choice(l + 1, size=n, p=fitness_prob)
    # Simulate mutations.
    mut_ben = np.random.poisson(lam=u_ben, size=n)
    mut_del = np.random.poisson(lam=u_del, size=n)
    return newborn, mut_ben, mut_del

def simple_landscape_wright_fisher(
    n: 'int, population size',
    l: 'int, size of the landscape, i.e., maximum number of mutations',
    u_ben: 'float, beneficial mutation rate',
    u_del: 'float, deleterious mutation rate',
    counts: 'numpy.array, each element corresponds to number of individuals',
    fitness: 'numpy.array, each element corresponds to fitness of each bin',
) -> 'numpy.array, counts for the next iteration':
    """
    Simulate one iteration of the Wright Fisher process on a simple fitness
    landscape, upon which individuals have 0 ~ l deleterious mutations.
    """
    newborn, mut_ben, mut_del = _generate_newborn_and_mutation(
        n, l, u_ben, u_del, counts, fitness,
    )
    counts_next = np.zeros(l + 1, dtype=int)
    for i in range(n):
        # Number of mutations are bounded by [0, l].
        counts_next[np.clip(
            newborn[i] - mut_ben[i] + mut_del[i],
            a_min=0,
            a_max=l,
        )] += 1
    return counts_next

def adjacent_landscape_wright_fisher(
    n: 'int, population size',
    l_l: 'int, size of the left fitness landscape',
    l_r: 'int, size of the right fitness landscape',
    u_ben: 'float, beneficial mutation rate',
    u_del: 'float, deleterious mutation rate',
    counts: 'numpy.array, each element corresponds to number of individuals',
    fitness: 'numpy.array, each element corresponds to fitness of each bin',
) -> 'numpy.array, the counts for the next iteration':
    """
    Simulate one iteration of the Wright Fisher process on two tail-to-tail
    adjacent fitness landscapes.
    """
    newborn, mut_ben, mut_del = _generate_newborn_and_mutation(
        n, l_l + l_r + 1, u_ben, u_del, counts, fitness,
    )
    counts_next = np.zeros(l_l + l_r + 2, dtype=int)
    for i in range(n):
        if newborn[i] <= l_l:
            # Within the range of the left landscape
            m = newborn[i] - mut_ben[i] + mut_del[i]
            if m > l_l:
                # Across the valley
                counts_next[l_l + 1] += 1
            else:
                counts_next[max(0, m)] += 1
        else:
            # Within the range of the right landscape
            m = newborn[i] + mut_ben[i] - mut_del[i]
            if m <= l_l:
                # Across the valley
                counts_next[l_l] += 1
            else:
                counts_next[min(l_l + l_r + 1, m)] += 1
    return counts_next

def hybrid_landscape_wright_fisher(
    n: 'int, population size',
    l_l: 'int, size of the left fitness landscape',
    l_r: 'int, size of the right fitness landscape',
    u_ben: 'float, beneficial mutation rate',
    u_del: 'float, deleterious mutation rate',
    p_r: 'float, probability to take a path to the right landscape',
    counts: 'numpy.array, each element corresponds to number of individuals',
    fitness: 'numpy.array, each element corresponds to fitness of each bin',
) -> 'numpy.array, the counts for the next iteration':
    """
    Simulate one iteration of the Wright Fisher process on two head-to-head
    hybrid fitness landscapes.
    """
    newborn, mut_ben, mut_del = _generate_newborn_and_mutation(
        n, l_l + l_r, u_ben, u_del, counts, fitness,
    )
    counts_next = np.zeros(l_l + l_r + 1, dtype=int)
    for i in range(n):
        if newborn[i] < l_l:
            # Within the range of the left landscape and off the peak
            m = newborn[i] + mut_ben[i] - mut_del[i]
            counts_next[max(0, min(l_l, m))] += 1
        elif newborn[i] > l_l:
            # Within the range of the right landscape and off the peak
            m = newborn[i] - mut_ben[i] + mut_del[i]
            counts_next[min(l_l + l_r, max(l_l, m))] += 1
        else:
            # At the peak
            m = max(0, mut_del[i] - mut_ben[i])
            counts_next[
                l_l + m if np.random.binomial(1, p_r) == 1
                else l_l - m
            ] += 1
    return counts_next