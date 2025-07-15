import numpy as np
import math
import galois
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from itertools import combinations_with_replacement, permutations, product
from typing import Dict, Tuple, List
from fractions import Fraction

def get_plot(p):
    if (p.shape)[1] == 2:
        df = pd.DataFrame(p, columns=["x", "y"])
        #print(df)
        sns.scatterplot(df, x="x", y="y").plot()
    elif (p.shape)[1] == 3:
        df = pd.DataFrame(p, columns=["x", "y", "z"])
        fig = plt.figure(figsize=(8, 6))
        fig = plt.figure(figsize=(8, 8))
        fig = px.scatter_3d(df, x='x', y='y', z='z')
        fig.show()

def e_param(t, m, s):
    n = t + s
    x = s
    if n < x:
        return None
    result = [1] * x
    remaining = n - x
    i = 0
    while remaining > 0:
        result[i] += 1
        remaining -= 1
        i = (i + 1) % x
    return result


def is_prime_power(n):
    if n < 2:
        return False
    for p in range(2, int(n ** 0.5) + 1):
        if is_prime(p):
            power = p
            while power <= n:
                if power == n:
                    return True
                power *= p
    return is_prime(n)


def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def print_generator_matrix(G, e):
    """
    Печатает матрицу с разметкой секций в стиле статьи.
    """
    m = G.shape[0]
    num_sections = (m + e - 1) // e

    print("Матрица:")
    for i in range(m):
        section = i // e + 1
        row_str = " ".join(str(G[i, j]) for j in range(m))
        print(f"{i:2d} | {row_str} | секция {section}")
    print()


def generate_excellent_poly(b, e, s):
    assert is_prime_power(b), "b must be a prime power"
    pi = []
    unique_polys = {degree: set() for degree in set(e)}

    for deg in set(e):
        all_irred = list(galois.irreducible_polys(b, deg))
        all_irred = [p for p in all_irred if p != galois.Poly([1, 0], field=galois.GF(b))]
        if len(all_irred) < e.count(deg):
            raise ValueError(f"Not enough irreducible polys {deg} over GF({b}) for {e.count(deg)} request (available: {len(all_irred)}).")
        unique_polys[deg] = set(all_irred)
    used = {deg: set() for deg in set(e)}
    for deg in e: 
        available = sorted(unique_polys[deg] - used[deg], key=lambda p: tuple(p.coeffs))
        P = available[0]  
        used[deg].add(P)
        pi.append(P)
    return pi


def generate_recurrent_sequence(poly, u, m):
    e = poly.degree
    degree = e * u
    poly_u = poly ** u
    coeffs = poly_u.coeffs 
    GF = poly.field
    alpha = [GF(0)] * (e * (u - 1))
    alpha += [GF(1)] + [GF(0)] * (degree - (e * (u - 1)) - 1)

    while len(alpha) < m + degree:
        acc = GF(0)
        for k in range(1, degree + 1):
            acc -= coeffs[k] * alpha[-k]
        alpha.append(acc)
    return alpha

def build_generator_matrix(poly, m):
    e = poly.degree
    num_sections = (m + e - 1) // e  # div(m-1, e) + 1
    G = np.zeros((m, m), dtype=int)

    for u in range(1, num_sections + 1):
        alpha = generate_recurrent_sequence(poly, u, m)
        r_h = e - 1 if u < num_sections else (m - 1) % e

        for r in range(r_h + 1):
            j = e * (u - 1) + r
            if j >= m:
                break
            for k in range(m):
                G[j, k] = int(alpha[r + k])
    return G


def generate_generator_matrices(b, t, m, s, verbose=100):
    e = e_param(t, m, s)
    assert e is not None, "Wrong (t, m, s)"

    pi_list = generate_excellent_poly(b, e, s)
    print(pi_list)
    matrices = []
    for i in range(s):
        G = build_generator_matrix(pi_list[i], m)
        matrices.append(G)
    
    return matrices


def rnum_opt(b, v):
    v = np.asarray(v) 
    m = v.shape[1] 
    powers = b ** np.arange(m-1, -1, -1) 
    print(powers)
    return np.dot(v, powers)


def vecbm_opt(b, m, n):
    n = np.asarray(n) 
    shape = n.shape
    n = n.ravel() 
    x = (n[:, None] // b**np.arange(m)) % b
    
    return x.reshape(*shape, m)


def get_points_opt(b, t, m, s, verbose=100):
    gf = galois.GF(b)
    G = generate_generator_matrices(b, t, m, s, verbose)
    if verbose==50:
        print(*G, sep="\n")
    n_values = np.arange(b**m)
    vecs = vecbm_opt(b, m, n_values)  # (b**m, m)
    G_gf = gf(G)  # (s, m, m)
    vecs_gf = gf((vecs.T)[::-1])  # (m, b**m)
    result = np.empty((s,m,b**m), dtype=object)
    for i in range(s):
        result[i] = G_gf[i] @ vecs_gf  
    powers = b ** np.arange(m-1, -1, -1) 
    rnums = np.tensordot(result, powers, axes=(1, 0))
    points = (rnums.T) * (b**(-m))  
    return points

def rosenbloom_tsfasman_net(q, m, s, beta=None):
    """
    Construction (0, m, s)-nets over GF(q) via Rosenbloom–Tsfasman method.
    """
    GF = galois.GF(q)

    if s > q:
        raise ValueError("Wrong s(s must be less or equal than q)")
    all_field_elements_gf = GF.elements 
    S_gf = all_field_elements_gf[:s]
    is_default_beta = False
    if beta is None:
        is_default_beta = True
    else:
        beta_keys = np.array(sorted(list(beta.keys())))
        if np.array_equal(beta_keys, np.arange(q)):
            beta_lookup_array = np.array([beta[i] for i in range(q)], dtype=np.float64)
            beta_map_func = lambda x_int_array: beta_lookup_array[x_int_array.astype(np.int64)] 
        else:
            _beta_vec = np.vectorize(lambda val_int: beta.get(val_int), otypes=[np.float64])
            beta_map_func = lambda x_int_array: _beta_vec(x_int_array)
    coeffs_int_tuples = product(range(q), repeat=m)
    poly_list = [galois.Poly(list(c_tuple)[::-1], field=GF) for c_tuple in coeffs_int_tuples]
    num_polynomials = q**m
    points = np.zeros((num_polynomials, s), dtype=np.float64)
    q_powers = q**(-(np.arange(m, dtype=np.float64) + 1.0))
    eval_matrix_int = np.zeros((s, m), dtype=np.int64) 
    for idx_f, f_poly in enumerate(poly_list):
        current_deriv_poly = f_poly
        for j_deriv_order in range(m):
            eval_results_gf = current_deriv_poly(S_gf)
            eval_matrix_int[:, j_deriv_order] = eval_results_gf.astype(np.int64)
            if j_deriv_order < m - 1:
                current_deriv_poly = current_deriv_poly.derivative()
        if is_default_beta:
            digits_matrix = eval_matrix_int.astype(np.float64)
        else:
            digits_matrix = beta_map_func(eval_matrix_int)      
        points[idx_f, :] = np.dot(digits_matrix, q_powers)
    return points

def get_points_opt_custom(G, b, m, s):
    gf = galois.GF(b)

    # Проверка размера G: ожидается (s, m, m)
    assert np.array(G).shape == (s, m, m), f"Ожидается форма G: ({s}, {m}, {m}), получено: {G.shape}"

    # Векторы в (b**m, m) представлении
    n_values = np.arange(b**m)
    vecs = vecbm_opt(b, m, n_values)  # (b**m, m)

    # Преобразование в GF(b)
    G_gf = gf(G)  # (s, m, m)
    vecs_gf = gf(vecs.T)  # (m, b**m), без реверса!

    # Умножение всех матриц G[i] на вектор vecs_gf
    result = np.empty((s, m, b**m), dtype=gf)
    for i in range(s):
        result[i] = G_gf[i] @ vecs_gf  # (m, b**m)

    # Преобразуем координаты обратно в числа
    powers = b ** np.arange(m - 1, -1, -1)  # (m,)
    rnums = np.tensordot(result, powers, axes=(1, 0))  # (s, b**m)

    # Преобразуем в точки на [0,1)^s
    points = (rnums.T) * (b ** -m)  # (b**m, s)

    return points

def generate_D_A_pairs(t: int, m: int, s: int, b: int) -> Dict[Tuple[int, ...], List[Tuple[int, ...]]]:
    """
    Генерирует словарь, где ключом является D, а значением возможные А.
    """
    n = m - t
    D_A_to_index = {}

    D_set = set()
    for D in combinations_with_replacement(range(n + 1), s):
        if sum(D) == n:
            D_set.update(permutations(D))


    for D in D_set:
        A_list = []
        A_ranges = [range(b ** d_i) for d_i in D]
        A_list.extend(product(*A_ranges))
        D_A_to_index[D] = A_list

    return D_A_to_index

def convert_points_to_fractions(points: np.ndarray, b: int, m: int) -> np.ndarray:
    """
    Возвращает numpy.ndarray: массив точек с координатами типа Fraction
    """
    points_fractions = np.empty(points.shape, dtype=object) 
    denominator = b ** m  # Используем b^m в качестве знаменателя

    for i, point in enumerate(points):
        for j, x in enumerate(point):
            # Пропускаем преобразование, если x уже является дробью
            if isinstance(x, Fraction):
                points_fractions[i, j] = x
                continue

            # Используем точные значения для 0 и 1
            if x < 1e-10:
                points_fractions[i, j] = Fraction(0, 1)
            elif x > 1 - 1e-10:
                points_fractions[i, j] = Fraction(1, 1)
            else:
                # Создаем дробь со знаменателем b^m
                numerator = int(round(x * denominator))
                points_fractions[i, j] = Fraction(numerator, denominator).limit_denominator()

    return points_fractions

def check_tms_network(points: np.ndarray, t: int, m: int, s: int, b: int) -> bool:
    """
    Проверяет, является ли набор точек (t,m,s)-сетью с основанием b.
    Возвращает True если points образуют (t,m,s)-сеть, False иначе
    """
    points_fractions = convert_points_to_fractions(points, b, m)
    D_A_to_index = generate_D_A_pairs(t, m, s, b)

    # Создаем словарь для быстрого поиска индексов
    D_A_indices = {D: {A: i for i, A in enumerate(A_list)}
                   for D, A_list in D_A_to_index.items()}

    unique_D = list(D_A_to_index.keys())
    num_D = len(unique_D)
    max_A_count = max(len(A_list) for A_list in D_A_to_index.values())

    counters = np.zeros((num_D, max_A_count), dtype=np.uint32)

    for point in points_fractions:
        for d_index, D in enumerate(unique_D):
            A = tuple(int(point[j] * (b ** D[j])) for j in range(s))

            if all(a < b ** d for a, d in zip(A, D)):
                if A in D_A_indices[D]:
                    a_index = D_A_indices[D][A]
                    counters[d_index, a_index] += 1

    return np.all(counters == b**t)

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
