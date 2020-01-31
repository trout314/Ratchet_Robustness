"""
Define various fitness landscapes.

Fitness landscape notations:
    L: genome size, i.e., maximum number of mutations
    L_l, L_r: size of the left and right fitness landscape, respectively
    k: number of mutations
    s: fitness effect of each mutation
    s_l, s_r: fitness effect of each mutation on the left and right landscape,
              respectively
    eps: epistasis
    eps_l, eps_r: epistasis on the left and right landscape, respectively
    
Simple linear landscape:
    fitness = exp(-s * k ^ (1 - eps))
        where 0 <= k <= L

Adjacent landscape:
    fitness =
        if 0 <= k <= L_l
            exp(-s_l * k ^ (1 - eps_l))
        elif L_l + 1 <= k <= (L_l + L_r + 1)
            exp(-s_r * (L_l + L_r + 1 - k) ^ (1 - eps_r))

Hybrid landscape:
    fitness = 
        if 0 <= k <= L_l
            exp(-s_l * (L_l - k) ^ (1 - eps_l))
        elif L_l + 1 <= k <= (L_l + L_r)
            exp(-s_r * (k - L_l) ^ (1 - eps_r))
"""
import numpy as np

def simple_landscape(
    s: 'float, mutational effect',
    eps: 'float, epistasis',
    k: 'int or numpy.array, number of mutations'
) -> 'float or numpy.array, fitness':
    """Calculate fitness = exp(-s * k ^ (1 - eps))."""
    return np.exp(-s * np.power(k, 1 - eps))

def adjacent_landscape(
    s_l: 'float, mutational effect on the left fitness landscape',
    s_r: 'float, mutational effect on the right fitness landscape',
    eps_l: 'float, epistasis of the left fitness landscape',
    eps_r: 'float, epistasis of the right fitness landscape',
    l_l: 'int, size of the left fitness landscape',
    l_r: 'int, size of the right fitness landscape',
    k: 'int or numpy.array, distance from the origin of the left landscape',
) -> 'float or numpy.array, fitness':
    """Combine two fitness landscapes tail-to-tail."""
    return np.array([
        np.exp(-s_l * np.power(d, 1 - eps_l)) if 0 <= d <= l_l
        else np.exp(-s_r * np.power(l_l + l_r + 1 - d, 1 - eps_r))
        for d in list(k)
    ])

def hybrid_landscape(
    s_l: 'float, mutational effect on the left fitness landscape',
    s_r: 'float, mutational effect on the right fitness landscape',
    eps_l: 'float, epistasis of the left fitness landscape',
    eps_r: 'float, epistasis of the right fitness landscape',
    l_l: 'int, size of the left fitness landscape',
    l_r: 'int, size of the right fitness landscape',
    k: 'int or numpy.array, distance from the boundary of the left landscape',
) -> 'float or numpy.array, fitness':
    """Combine two fitness landscapes head-to-head."""
    return np.array([
        np.exp(-s_l * np.power(l_l - d, 1 - eps_l)) if 0 <= d <= l_l
        else np.exp(-s_r * np.power(d - l_l, 1 - eps_r))
        for d in list(k)
    ])
