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

def generate_D_vectors(t: int, m: int, s: int) -> np.ndarray:
    """
    Generates all possible integer vectors D = (d_1, ..., d_s) for a (t, m, s)-net.
    These vectors satisfy d_i >= 0 and sum(d_i) = m - t, and they define the
    elementary intervals used for the verification check.
    """
    n = m - t
    D_list = []

    for D in combinations_with_replacement(range(n + 1), s):
        if sum(D) == n:
            D_list.extend(set(permutations(D)))

    D_array = np.array(D_list)
    D_array = np.unique(D_array, axis=0)
    
    return D_array

def generate_D_A_pairs(t: int, m: int, s: int, b: int) -> Dict[Tuple[int, ...], List[Tuple[int, ...]]]:
    """
    Generates a dictionary that maps each vector D to a list of all possible
    corresponding vectors A. For a given D=(d_1, ..., d_s), a vector A is (a_1, ..., a_s) where 0 <= a_j < b^d_j for each j.
    """
    D_array = generate_D_vectors(t, m, s)

    D_A_to_index = {}
    for D in D_array:
        A_list = [] 
        A_ranges = [range(b ** d_i) for d_i in D]
        for A in product(*A_ranges):
            A_list.append(A) 

        D_A_to_index[tuple(D)] = A_list 
    return D_A_to_index

def convert_points_to_fractions(points: np.ndarray, b: int) -> np.ndarray:
    """
    Converts an array of floating-point coordinates to an array of Fraction objects.
    This is crucial for exact arithmetic when checking if points fall into b-adic intervals.
    """
    points_fractions = []
    for point in points:
        point_fractions = []
        for x in point:
            if isinstance(x, Fraction):
                point_fractions.append(x)
            else:
                x_str = f"{x:.20f}"
                x_str = x_str.rstrip('0').rstrip('.')
                x_float = float(x_str)
                if x_float < 1e-10:
                    point_fractions.append(Fraction(0, 1))
                elif x_float > 1 - 1e-10:
                    point_fractions.append(Fraction(1, 1))
                else:

                    found = False
                    for n in range(1, 15):
                        denominator = b**n
                        numerator = int(round(x_float * denominator))
                        if abs(x_float - numerator / denominator) < 1e-10:
                            point_fractions.append(Fraction(numerator, denominator))
                            found = True
                            break
                    if not found:
                        point_fractions.append(Fraction(x_str))
        points_fractions.append(point_fractions)
    return np.array(points_fractions)

def check_tms_network(points: np.ndarray, t: int, m: int, s: int, b: int) -> bool:
    """
    Checks whether the set of points forms a (t,m,s)-net in base b.
    Returns True if the points form a (t,m,s)-net, False otherwise.

    :param points: An array of points to be checked for the (t, m, s)-net property.
    :param t: The quality parameter t.
    :param m: The number of points in the net is b^m.
    :param s: The dimension of the net.
    :param b: The base of the (t, m, s)-net.
    """

    if points.shape[0] != b**m:
        print(f"Error: Expected {b**m} points, but got {points.shape[0]}.")
        return False
        
    points_fractions = convert_points_to_fractions(points, b)
    
    D_A_to_index = generate_D_A_pairs(t, m, s, b)
    
    unique_D = list(D_A_to_index.keys())
    for D in unique_D:
        for A in D_A_to_index[D]:
            count = 0
            for point in points_fractions:
                is_in_interval = True
                for j in range(s):
                    # Check if the point lies within the interval [a_j/b^d_j, (a_j+1)/b^d_j)
                    if not (Fraction(A[j], b**D[j]) <= point[j] < Fraction(A[j] + 1, b**D[j])):
                        is_in_interval = False
                        break
                if is_in_interval:
                    count += 1

            if count != b**t:
                print(f"Check failed for interval D={D}, A={A}. Found {count} points, expected {b**t}.")
                return False
                
    return True

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

    def verify(self) -> bool:
        """
        Checks whether a set of points is a (t,m,s)-network.If the check with the initial 't' fails,
        the function finds the smallest possible value of t for which the points form a valid network and updates self.t.
        """
        print(f"--- Starting verification for t = {self.t}... ---")

        if self.points.shape[0] != self.b**self.m:
            print(f"Critical error: For a (t, m, s)-network of radix {self.b} with m={self.m} there should be {self.b**self.m} points, but {self.points.shape[0]} is provided.")
            print("Verification interrupted.")
            return False

        points_fractions = convert_points_to_fractions(self.points, self.b)

        best_t_found = -1
        for new_t in range(self.m + 1):
            if _is_tms_network(points_fractions, new_t, self.m, self.s, self.b):
                best_t_found = new_t
                break

        if best_t_found == -1:
            print("Verification failed: No suitable value for t could be found.")
            return False
        if best_t_found == self.t:
            print(f"Verification is successful: the points indeed form a ({self.t}, {self.m}, {self.s})-network.")
            return True
        else:
            old_t = self.t
            self.t = best_t_found
            print(f"Initial check for t={old_t} failed.")
            print(f"---Updating---")
            print(f"The best value of t for a given set of points is found: t = {self.t}.")
            print(f"The object's t parameter was updated from {old_t} to {self.t}.")
            return True


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
        Constructs a (0, m, s)-network using the Rosenblum-Tsfasman method.Due to its combinatorial nature, this method takes a long time to work.

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
