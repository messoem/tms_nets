import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import math

n = 2003
s = 6
hidden_size = 512
lr = 3e-4
num_episodes = 200
gamma_disc = 0.99
ppo_epochs = 10
clip_eps = 0.2
entropy_coef = 0.03
value_coef = 0.5
device = torch.device("cpu")

def b2(frac):
    return frac*frac - frac + 1/6.0

def compute_e2(current_z, n, gamma):
    if len(current_z) == 0:
        return 0.0
    sum_prod = 0.0
    z_arr = torch.tensor(current_z, dtype=torch.long, device=device)
    for k in range(n):
        fracs = (k * z_arr % n) / n
        prod = torch.prod(1.0 + gamma * 12.0 * b2(fracs))
        sum_prod += prod.item()
    constant = (1.0 + gamma / 3.0) ** len(current_z)
    return -constant + sum_prod / n

best_e2 = float('inf')
best_z = None

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(s + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, n-1)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state):
        shared = self.shared(state)
        return self.actor(shared), self.critic(shared)

policy = ActorCritic().to(device)
optimizer = optim.Adam(policy.parameters(), lr=lr)

def get_state(step, current_z):
    padding = [0.0] * (s - len(current_z))
    state_list = [step / s] + [zj / n for zj in current_z] + padding
    return torch.tensor(state_list, dtype=torch.float32, device=device).unsqueeze(0)

for episode in range(num_episodes):
    states = []
    actions_idx = []
    log_probs = []
    values = []
    rewards = []                
    current_z = []
    prev_e2 = 0.0

    for step in range(s):
        state = get_state(step, current_z)
        states.append(state)

        logits, value = policy(state)
        dist = Categorical(logits=logits)

        action_idx = dist.sample()
        z_value = action_idx.item() + 1

        actions_idx.append(action_idx)
        log_probs.append(dist.log_prob(action_idx))
        values.append(value)

        current_z.append(z_value)

        current_e2 = compute_e2(current_z, n, gamma_disc)
        delta = prev_e2 - current_e2
        rewards.append(delta)
        prev_e2 = current_e2

    final_e2 = prev_e2
    if final_e2 < best_e2:
        best_e2 = final_e2
        best_z = current_z.copy()
        print(f"Episode {episode:7d} | BEST E² = {best_e2:.12f} | z = {current_z}")

    _, next_value = policy(get_state(s, current_z))
    returns = []
    G = next_value.item()
    for r in reversed(rewards):
        G = r + gamma_disc * G
        returns.insert(0, G)

    returns = torch.tensor(returns, device=device).unsqueeze(1)
    values = torch.cat(values)
    advantages = returns - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    old_log_probs = torch.cat(log_probs).detach()

    for _ in range(ppo_epochs):
        new_logits_list = [policy(states[i])[0] for i in range(s)]
        new_logits = torch.cat(new_logits_list)
        new_values = torch.cat([policy(states[i])[1] for i in range(s)])

        dist_new = Categorical(logits=new_logits)
        new_log_probs = dist_new.log_prob(torch.cat(actions_idx))

        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(new_values, returns)
        entropy = dist_new.entropy().mean()

        loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

print("\n=== FINAL BEST ===")
print("BEST E²:", best_e2)
print("BEST z :", best_z)

import numpy as np
import math
import time
import scipy.stats.qmc as qmc
from torch.quasirandom import SobolEngine
rl_z = best_z

def generate_lattice_points(z: list, n: int):
    """Генерирует n точек rank-1 lattice rule по вектору z"""
    z = np.array(z, dtype=int)
    points = np.zeros((n, len(z)))
    for k in range(n):
        points[k] = (k * z % n) / n  
    return points

print("n = ", n, "s = ", s)
print("=== RL vector ===")
rl_points = generate_lattice_points(rl_z, n)
rl_disc = qmc.discrepancy(rl_points)
print(f"RL z = {rl_z}")
print(f"star-discrepancy ≈ {rl_disc:.7f}\n")

print("=== sobol seq ===")
soboleng = torch.quasirandom.SobolEngine(dimension=4)
points = soboleng.draw(997)
print(f"sobol seq")
print(f"star-discrepancy ≈ {qmc.discrepancy(points):.7f}\n")
print("=== random points ===")
rnd_points = np.random.rand(997,4)
print(f"random points")
print(f"star-discrepancy ≈ {qmc.discrepancy(rnd_points):.7f}\n")

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.qmc as qmc
import torch
from torch.quasirandom import SobolEngine

n_values = np.arange(10, 2001, 10)

results_rl = []
results_sobol = []
results_random = []

soboleng = SobolEngine(dimension=s)
max_n = n_values[-1]
sobol_all_points = soboleng.draw(int(max_n))

print(f"Начинаем расчет discrepancy для размерности s={s}...")

for n in n_values:
    points_rl = generate_lattice_points(rl_z, n)
    disc_rl = qmc.discrepancy(points_rl)
    results_rl.append(disc_rl)
    
    points_sobol = sobol_all_points[:n, :].numpy()
    disc_sobol = qmc.discrepancy(points_sobol)
    results_sobol.append(disc_sobol)
    
    points_rnd = np.random.rand(n, s)
    disc_rnd = qmc.discrepancy(points_rnd)
    results_random.append(disc_rnd)
    
    if n % 200 < 20: 
        print(f"Processed n = {n}")

plt.figure(figsize=(10, 6))

plt.plot(n_values, results_rl, label='RL Lattice (best_z)', marker='o', markersize=3)
plt.plot(n_values, results_sobol, label='Sobol Sequence', marker='s', markersize=3)
plt.plot(n_values, results_random, label='Random (MC)', marker='x', markersize=3, alpha=0.5)

plt.title(f'Star Discrepancy vs Number of Points (Dim={s})')
plt.xlabel('Number of Points (n)')
plt.ylabel('Star Discrepancy')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)

plt.yscale('log')
plt.xscale('log')

plt.tight_layout()
plt.show()
