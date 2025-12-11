import numpy as np
import itertools
import galois
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


def rank_GF(matrix_gf):
    return np.linalg.matrix_rank(matrix_gf)

def compute_t(C_matrices_np, b):
    try:
        GF = galois.GF(b)
    except (LookupError, ValueError):
        GF = galois.GF(b)
        
    C_matrices = [GF(mat.astype(int)) for mat in C_matrices_np]
    
    s = len(C_matrices)
    n_rows, m_cols = C_matrices[0].shape
    start_d = min(n_rows * s, m_cols) 

    for d in range(start_d, 0, -1):
        all_combinations_full_rank = True
        max_rows_per_matrix = min(n_rows, d)
        
        for combo in itertools.product(range(max_rows_per_matrix + 1), repeat=s):
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
            
            if combined_matrix.shape[0] > combined_matrix.shape[1]:
                 all_combinations_full_rank = False
                 break
            
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


class MatrixExtensionEnv(gym.Env):
    def __init__(self, A_tensor, b):
        super(MatrixExtensionEnv, self).__init__()
        
        self.A_tensor = A_tensor
        self.b = b
        self.s, self.m, _ = A_tensor.shape
        self.m1 = self.m + 1
        self.total_steps = self.s * self.m1
        
        self.action_space = spaces.Discrete(self.b)
        self.observation_space = spaces.Box(low=-1, high=self.b-1, 
                                            shape=(self.total_steps,), dtype=np.float32)
        
        self.current_step = 0
        self.state = np.full(self.total_steps, -1, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = np.full(self.total_steps, -1, dtype=np.float32)
        return self.state, {}

    def step(self, action):
        self.state[self.current_step] = action
        self.current_step += 1
        
        terminated = False
        truncated = False
        reward = 0.0
        info = {}
        
        if self.current_step >= self.total_steps:
            terminated = True
            final_solution = self.state.astype(int).copy()
            info["final_solution"] = final_solution
            generators = construct_extended_matrices(self.A_tensor, final_solution, self.s, self.m, self.m1)
            t_val = compute_t(generators, self.b)
            reward = (self.m1 - t_val) ** 2
            if t_val == 0:
                reward += 10.0
            
            info["t_val"] = t_val
            
        return self.state, reward, terminated, truncated, info

class SaveBestModelCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(SaveBestModelCallback, self).__init__(verbose)
        self.best_reward = -np.inf
        self.best_vector = None
        self.best_t = float('inf')

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals["dones"]):
            if done:
                reward = self.locals["rewards"][i]
                info = self.locals["infos"][i]
                
                if reward > self.best_reward:
                    self.best_reward = reward
                    
                    if "final_solution" in info:
                        self.best_vector = info["final_solution"]
                        if "t_val" in info:
                            current_t = info["t_val"]
                            if self.verbose > 0:
                                print(f"New best found! Reward: {reward:.2f}, t: {current_t}")
        return True


def solve_with_rl(matricies, b=2, total_timesteps=5000):
    A_tensor = np.array(matricies)
    env = MatrixExtensionEnv(A_tensor, b)
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.001, ent_coef=0.01)
    callback = SaveBestModelCallback(verbose=1)
    
    print(f"Starting RL training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    if callback.best_vector is not None:
        if np.any(callback.best_vector < 0):
            print("Error: Saved vector contains invalid values!")
            return None, None

        final_gens = construct_extended_matrices(A_tensor, callback.best_vector, env.s, env.m, env.m1)
        final_t = compute_t(final_gens, b)
        
        print("\nOptimization finished.")
        print(f"Best t found: {final_t}")
        return final_gens, final_t
    else:
        print("No solution found within timesteps.")
        return None, None


final_gens, t_param = solve_with_rl(matricies, b=b, total_timesteps=10000)

if final_gens is not None:
    print("Extended Matrices:")
    for i, mat in enumerate(final_gens):
        print(f"Dim {i}:\n{mat}")
