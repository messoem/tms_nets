import numpy as np
import math
import galois
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from itertools import product

class TMSNet:
    """
    A class for storing, analyzing, and visualizing (t,m,s)-networks.

    :param b: base of (t, m, s)-net. Using for calculation over GF(b).
    :param t: quality measure of (t, m, s)-net.
    :param m: the number of points in the (t, m, s)-net is characterized as b^m
    :param s: dimension of (t, m, s)-net
    """
    def __init__(self, t, m, s, b, points):
        """
        Initializes an instance of the (t,m,s)-network.
        """
        self.t = t
        self.m = m
        self.s = s
        self.b = b
        self.points = points
        
        print(f"A ({self.t}, {self.m}, {self.s})-network is created on base {self.b}, containing {self.points.shape[0]} points.")

    def visualize(self):
        """
        Visualizes network points.
        Supports both 2D and 3D cases.
        """
        if self.points.shape[1] < 2:
            print("Visualization is only possible for s >= 2.")
            return
            
        if self.points.shape[1] == 2:
            df = pd.DataFrame(self.points, columns=["x", "y"])
            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=df, x="x", y="y", s=20).set_title(f'({self.t}, {self.m}, {self.s})-сеть')
            plt.grid(True)
            plt.show()
            
        elif self.points.shape[1] == 3:
            df = pd.DataFrame(self.points, columns=["x", "y", "z"])
            fig = px.scatter_3d(df, x='x', y='y', z='z', title=f'({self.t}, {self.m}, {self.s})-сеть')
            fig.update_traces(marker=dict(size=3))
            fig.show()
        else:
            print(f"Visualization for s={self.s} is not supported. Only the first 2 dimensions will be shown.")
            df = pd.DataFrame(self.points[:, :2], columns=["x", "y"])
            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=df, x="x", y="y", s=20).set_title(f'({self.t}, {self.m}, {self.s})-сеть (первые 2 измерения)')
            plt.grid(True)
            plt.show()


class PolynomialNetConstructor:
    """
    A class implementing algorithms for constructing (t,m,s)-networks using polynomials over finite fields.

    Current implementation include Niederreiter algorithm and Rosenbloom-Tsfasman algorithm.
    """

    @staticmethod
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n ** 0.5) + 1, 2):
            if n % i == 0: return False
        return True

    @staticmethod
    def is_prime_power(n):
        if n < 2: return False
        if PolynomialNetConstructor.is_prime(n): return True
        for p in range(2, int(n ** 0.5) + 1):
            if PolynomialNetConstructor.is_prime(p):
                power = p
                while power <= n:
                    if power == n: return True
                    if n % power != 0: break
                    power *= p
        return False

    @staticmethod
    def _e_param(t, m, s):
        n = t + s
        x = s
        if n < x: return None
        result = [1] * x
        remaining = n - x
        i = 0
        while remaining > 0:
            result[i] += 1
            remaining -= 1
            i = (i + 1) % x
        return result

    @staticmethod
    def _generate_excellent_poly(b, e, s):
        assert PolynomialNetConstructor.is_prime_power(b), "b must be a prime power"
        pi = []
        unique_polys = {degree: set() for degree in set(e)}

        for deg in set(e):
            all_irred = list(galois.irreducible_polys(b, deg))
            all_irred = [p for p in all_irred if p != galois.Poly([1, 0], field=galois.GF(b))]
            if len(all_irred) < e.count(deg):
                raise ValueError(f"There are not enough irreducible polynomials of degree {deg} in GF({b}) for {e.count(deg)} queries ({len(all_irred)} available).")
            unique_polys[deg] = set(all_irred)

        used = {deg: set() for deg in set(e)}

        for deg in e: 
            available = sorted(unique_polys[deg] - used[deg], key=lambda p: tuple(p.coeffs))
            P = available[0] 
            used[deg].add(P)
            pi.append(P)
        return pi

    @staticmethod
    def _generate_recurrent_sequence(poly, u, m):
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

    @staticmethod
    def _build_generator_matrix(poly, m):
        e = poly.degree
        num_sections = (m + e - 1) // e  
        G = np.zeros((m, m), dtype=int)
        for u in range(1, num_sections + 1):
            alpha = PolynomialNetConstructor._generate_recurrent_sequence(poly, u, m)
            r_h = e - 1 if u < num_sections else (m - 1) % e
            for r in range(r_h + 1):
                j = e * (u - 1) + r
                if j >= m:
                    break
                for k in range(m):
                    G[j, k] = int(alpha[r + k])
        return G

    @staticmethod
    def _generate_generator_matrices(b, t, m, s, verbose=False):
        e = PolynomialNetConstructor._e_param(t, m, s)
        assert e is not None, "Wrong params: t, m, s"
        assert t <= m, "t must be less or equal m"
        pi_list = PolynomialNetConstructor._generate_excellent_poly(b, e, s)
        if verbose:
            print("list of polys:", pi_list)
        matrices = []
        for i in range(s):
            G = PolynomialNetConstructor._build_generator_matrix(pi_list[i], m)
            matrices.append(G)
        
        return matrices

    @staticmethod
    def _vecbm_opt(b, m, n):
        """
        Converts an array of numbers n to their b-ary representations of fixed length m

        :param b: base of (t, m, s)-net. Using for calculation over GF(b).
        :param m: the number of points in the (t, m, s)-net is characterized as b^m
        """
        n = np.asarray(n) 
        shape = n.shape
        n = n.ravel()
        x = (n[:, None] // b**np.arange(m)) % b

        return x.reshape(*shape, m)

    @staticmethod
    def construct_niederreiter(b, t, m, s, verbose=False):
        """
        Constructing (t, m, s)-net via Niederreiter algorithm

        :param b: base of (t, m, s)-net. Using for calculation over GF(b).
        :param t: quality measure of (t, m, s)-net.
        :param m: the number of points in the (t, m, s)-net is characterized as b^m
        :param s: dimension of (t, m, s)-net
        """
        gf = galois.GF(b)
        G = PolynomialNetConstructor._generate_generator_matrices(b, t, m, s, verbose)
        if verbose:
            for i, matrix in enumerate(G):
                print(f"generating matricies G_{i+1}:\n{matrix}\n")
                
        n_values = np.arange(b**m)
        vecs = PolynomialNetConstructor._vecbm_opt(b, m, n_values)  
        G_gf = gf(G)
        vecs_gf = gf((vecs.T)[::-1])
        result = np.empty((s,m,b**m), dtype=object)
        for i in range(s):
            result[i] = G_gf[i] @ vecs_gf  
        powers = b ** np.arange(m-1, -1, -1) 
        rnums = np.tensordot(result, powers, axes=(1, 0))
        points = (rnums.T) * (b**(-m))  
        return points

    

    @staticmethod
    def construct_rosenbloom_tsfasman(q, m, s, beta=None):
        """
        Constructs a (0, m, s)-network using the Rosenblum-Tsfasman method.

        Due to its combinatorial nature, this method takes a long time to work.
        :param q: base of (0, m, s)-net. Using for calculation over GF(q).
        :param m: the number of points in the (0, m, s)-net is characterized as q^m
        :param s: dimension of (0, m, s)-net
        """
        GF = galois.GF(q)

        if s > q:
            raise ValueError("s must be least or equal q")
            
        S_gf = GF.elements[:s]
        is_default_beta = beta is None

        if not is_default_beta:
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
