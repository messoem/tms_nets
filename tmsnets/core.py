import numpy as np
import math
import galois
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def get_plot(p):
    if (p.shape)[1] == 2:
        df = pd.DataFrame(p, columns=["x", "y"])
        #print(df)
        sns.scatterplot(df, x="x", y="y").plot()
    elif (p.shape)[1] == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        df = pd.DataFrame(p, columns=["x", "y", "z"])
        ax.scatter(df['x'], df['y'], df['z'])
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.title('3D Scatter Plot with Seaborn')
        plt.show()


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
            raise ValueError(f"Недостаточно неприводимых многочленов степени {deg} в GF({b}) для {e.count(deg)} запросов (доступно {len(all_irred)}).")
        unique_polys[deg] = set(all_irred)
    
    used = {deg: set() for deg in set(e)}

    for i in range(s):
        deg = e[i]
        available = unique_polys[deg] - used[deg]
        P = available.pop()
        used[deg].add(P)
        pi.append(P)
    
    return pi


def generate_recurrent_sequence(poly, u, m):
    """
    Генерация линейной рекуррентной последовательности α для секции u с характеристическим многочленом poly^u.
    """
    e = poly.degree
    degree = e * u

    poly_u = poly ** u
    coeffs = poly_u.coeffs[::-1]

    # Определяем поле
    GF = poly.field

    # Начальные элементы: e*(u-1) нулей и хотя бы одна единица
    alpha = [GF(0)] * (e * (u - 1))
    alpha += [GF(1)] * (degree - (e * (u - 1)))

    while len(alpha) < m + degree:
        acc = GF(0)
        for k in range(degree):
            acc += coeffs[k] * alpha[-degree + k]
        alpha.append(acc)
    return alpha


def build_generator_matrix(poly, m):
    """
    Построение одной матрицы Γ[i] для заданного многочлена и параметра m.
    """
    e = poly.degree
    num_sections = math.ceil(m / e)  # div(m-1, e) + 1
    
    # Пустая матрица m x m над F_b
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
    """
    Основная функция для генерации всех s генерирующих матриц Γ[1], ..., Γ[s]
    """
    e = e_param(t, m, s)
    assert e is not None, "Некорректные параметры t, m, s"

    pi_list = generate_excellent_poly(b, e, s)
    if verbose==100:
        print(pi_list)
    matrices = []
    for i in range(s):
        G = build_generator_matrix(pi_list[i], m)
        matrices.append(G)
    
    return matrices


def rnum_opt(b, v):
    v = np.asarray(v)  # Преобразуем в массив NumPy
    m = v.shape[1]  # Количество разрядов
    powers = b ** np.arange(m-1, -1, -1) 
    print(powers)
    return np.dot(v, powers)


def vecbm_opt(b, m, n):
    """ Преобразует массив чисел n в их b-ичные представления фиксированной длины m """
    n = np.asarray(n)  # Поддержка как скаляра, так и массива
    shape = n.shape  # Запоминаем форму входных данных
    n = n.ravel()  # Делаем одномерным (если был массив)
    
    # Вычисляем b-ичное представление сразу для всех элементов
    x = (n[:, None] // b**np.arange(m)) % b
    
    return x.reshape(*shape, m)


def get_points_opt(b, t, m, s, verbose=100):
    gf = galois.GF(b)
    G = generate_generator_matrices(b, t, m, s, verbose)
    if verbose==100:
        print(*G, sep="\n")
    n_values = np.arange(b**m)
    vecs = vecbm_opt(b, m, n_values)  # (b**m, m)
    G_gf = gf(G)  # (s, m, m)
    vecs_gf = gf((vecs.T)[::-1])  # (m, b**m)
    result = np.empty((s,m,b**m), dtype=object)
    for i in range(s):
        result[i] = G_gf[i] @ vecs_gf  # Умножаем в GF(b)
    powers = b ** np.arange(m-1, -1, -1)  # (m,) - степени b с конца
    rnums = np.tensordot(result, powers, axes=(1, 0))# (s, b**m)
    points = (rnums.T) * (b**(-m))  # Транспонируем (b**m, s) 
    return points
