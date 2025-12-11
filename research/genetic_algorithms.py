import numpy as np
import itertools
from deap import base, creator, tools, algorithms
from random import randint
from itertools import combinations

def compute_digits_matrix1(N, b, m1):

    powers = (b ** np.arange(m1)).reshape(1, -1)
    return np.arange(N).reshape(-1, 1) // powers % b  # (N, m1)

def partition_integer1(n, k):
    if k == 1:
        yield (n,)
    else:
        for i in range(n + 1):
            for tail in partition_integer1(n - i, k - 1):
                yield (i,) + tail

def compute_t_for_generators1(generators, b):

    s, m1, _ = generators.shape
    m = m1 - 1
    N = b ** m1

    digits = compute_digits_matrix1(N, b, m1)

    # y_list[dim] имеет форму (N, m1)
    y_list = [digits @ generators[dim].T % b for dim in range(s)]

    for t in range(0, m + 1):
        L = m1 - t
        for d_tuple in partition_integer1(L, s):
            indices = np.zeros((N, s), dtype=int)
            for dim in range(s):
                d = d_tuple[dim]
                if d == 0:
                    continue
                y = y_list[dim]  # (N, m1)
                weights = b ** np.arange(d - 1, -1, -1)
                indices[:, dim] = (y[:, :d] * weights).sum(axis=1)
            expected_cells = np.prod([b ** d for d in d_tuple])
            keys, counts = np.unique(indices, axis=0, return_counts=True)
            if len(keys) != expected_cells:
                continue
            if not np.all(counts == b ** t):
                continue
            return t
    return None    

def rank_GF(A):
    """
    Вычисляет ранг матрицы A над конечным полем, используя row-reduction.
    """
    R = A.row_reduce()
    return sum(not np.all(row == 0) for row in R)

def compute_t(C_matrices, b):
    GF = galois.GF(b)
    s = len(C_matrices)
    n, m = C_matrices[0].shape

    for C in C_matrices:
        assert C.shape == (n, m)

    for d in range(m, 0, -1):
        all_good = True
        for combo in itertools.product(range(n + 1), repeat=s):
            if sum(combo) != d:
                continue

            rows = []
            for j in range(s):
                rows.extend(C_matrices[j][:combo[j], :])
            A = GF(np.vstack(rows))
            rank = rank_GF(A)
            if rank < d:
                all_good = False
                break
        if all_good:
            return m - d
    return m


    
def optimize_last_columns_fixed_blocks(A_tensor, b=2, max_gen=50, pop_size=50):
    """
    Оптимизирует последние столбцы генераторных матриц (s x (m+1) x (m+1))
    с фиксированными начальными блоками A_tensor (s x m x m).
    Возвращает генераторы и минимальный t-параметр.
    """
    s, m, _ = A_tensor.shape
    m1 = m + 1
    N = b ** m1

    generators = np.zeros((s, m1, m1), dtype=int)
    generators[:, :m, :m] = A_tensor

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_int", randint, 0, b - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=s * m1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_indiv(ind):
        for dim in range(s):
            generators[dim, :, m] = ind[dim * m1 : (dim + 1) * m1]
        t_val = compute_t(generators, b=b)
        return (t_val if t_val is not None else m1,)

    toolbox.register("evaluate", eval_indiv)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=b - 1, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=max_gen, halloffame=hof, verbose=False)

    best_ind = hof[0]
    for dim in range(s):
        generators[dim, :, m] = best_ind[dim * m1 : (dim + 1) * m1]
    best_t = compute_t(generators, b=b)
    return generators, best_t
