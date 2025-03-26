import numpy as np
import galois


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
    alpha += [GF(1)] + [GF(0)] * (degree - (e * (u - 1)) - 1)

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
    num_sections = (m + e - 1) // e  # div(m-1, e) + 1
    
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

def vecbm(b, m, n):
    x = []
    while n >= b:
        x.append(n%b)
        n //= b
    x.append(n)
    if len(x) < m:
        while len(x) != m:
            x.append(0)
    return np.array(x[:m])

print(vecbm(3, 4, 32))

def rnum(b, v):
    res = 0
    m = len(v)
    for k in range(m):
        res += v[m-k-1]*b**k
    return res



