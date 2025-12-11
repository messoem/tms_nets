import numpy as np
import itertools
import math
import random
import galois
import copy

def rank_GF(matrix_gf):
    return np.linalg.matrix_rank(matrix_gf)

def compute_t(C_matrices_np, b):
    GF = galois.GF(b)
    C_matrices = [GF(mat.astype(int)) for mat in C_matrices_np]
    
    s = len(C_matrices)
    n_rows, m_cols = C_matrices[0].shape

    start_d = min(n_rows * s, m_cols) 

    for d in range(start_d, 0, -1):
        all_combinations_full_rank = True
        
        for combo in itertools.product(range(n_rows + 1), repeat=s):
            if sum(combo) != d:
                continue

            rows_list = []
            for j in range(s):
                count = combo[j]
                if count > 0:
                    rows_list.append(C_matrices[j][:count, :])
            
            if not rows_list:
                continue
                
            combined_matrix = np.vstack(rows_list)
            r = rank_GF(combined_matrix)
            
            if r < d:
                all_combinations_full_rank = False
                break
        
        if all_combinations_full_rank:
            return m_cols - d

    return m_cols

def construct_extended_matrices(A_tensor, last_columns_vector, s, m, m1):
    generators = np.zeros((s, m1, m1), dtype=int)
    generators[:, :m, :m] = A_tensor
    
    for dim in range(s):
        col_values = last_columns_vector[dim * m1 : (dim + 1) * m1]
        generators[dim, :, m] = col_values
        
    return generators

def simulated_annealing_extension(A_tensor, b=2, 
                                  initial_temp=10.0, 
                                  final_temp=0.01, 
                                  alpha=0.95, 
                                  iters_per_temp=50):
    s, m, _ = A_tensor.shape
    m1 = m + 1
    
    state_length = s * m1
    current_state = np.random.randint(0, b, size=state_length)
    
    gens = construct_extended_matrices(A_tensor, current_state, s, m, m1)
    current_energy = compute_t(gens, b)
    
    best_state = current_state.copy()
    best_energy = current_energy
    
    current_temp = initial_temp
    
    while current_temp > final_temp:
        for _ in range(iters_per_temp):
            neighbor_state = current_state.copy()
            
            idx_to_change = random.randint(0, state_length - 1)
            new_val = random.randint(0, b - 1)
            while new_val == neighbor_state[idx_to_change] and b > 1:
                 new_val = random.randint(0, b - 1)
            neighbor_state[idx_to_change] = new_val
            
            gens_neighbor = construct_extended_matrices(A_tensor, neighbor_state, s, m, m1)
            neighbor_energy = compute_t(gens_neighbor, b)
            
            delta = neighbor_energy - current_energy
            
            if delta < 0:
                current_state = neighbor_state
                current_energy = neighbor_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_state = current_state.copy()
            else:
                probability = math.exp(-delta / current_temp)
                if random.random() < probability:
                    current_state = neighbor_state
                    current_energy = neighbor_energy
        
        current_temp *= alpha
        
    final_generators = construct_extended_matrices(A_tensor, best_state, s, m, m1)
    return final_generators, best_energy

A_tensor_input = np.array(matricies) 
final_gens, t_param = simulated_annealing_extension(
    A_tensor_input, 
    b=b, 
    initial_temp=5.0,
    final_temp=0.05,
    alpha=0.90,
    iters_per_temp=20
)

print(f"Resulting t-parameter: {t_param}")
for i, mat in enumerate(final_gens):
    print(f"Dim {i}:\n{mat}")
