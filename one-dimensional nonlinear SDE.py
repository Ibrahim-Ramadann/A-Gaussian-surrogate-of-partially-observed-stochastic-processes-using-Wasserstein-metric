import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# 1D example: f(x) = sin(x)
# Simulate 10 runs comparing:
#  - True nonlinear SDE: dx = sin(x) dt + L dB
#  - Linear SDE surrogate: dx_lin = (A(t) x_lin + b(t)) dt + L dB  (A,b from q=Normal(mu,P))
#  - Variational mean mu(t) and variance P(t) ( propagation)
# ----------------------

np.random.seed(0)

# Time settings
dt = 0.01
T = 10.0
N = int(T / dt)
time = np.linspace(0, T, N + 1)

# Model
def f(x):
    return np.sin(x)

L = 0.3      # diffusion coefficient (scalar)
Q = 1.0      # Brownian covariance (scalar)

# Initial conditions
x0 = 0.0
mu0 = 0.0
P0 = 0.1

n_runs = 5

true_trajs = []
lin_trajs = []
mu_trajs = []
P_trajs = []

for run in range(n_runs):
    rng = np.random.default_rng(seed=run)  # per-run RNG for reproducibility

    x_true = np.zeros(N + 1)
    x_lin = np.zeros(N + 1)
    mu = np.zeros(N + 1)
    P = np.zeros(N + 1)

    x_true[0] = x0
    x_lin[0] = x0   # start linear sim from same initial sample
    mu[0] = mu0
    P[0] = P0

    for i in range(N):
        # Brownian increment (same for true and linear sim for fair comparison)
        dB = rng.normal(0.0, np.sqrt(dt * Q))

        # --- True nonlinear SDE (Euler-Maruyama) ---
        x_true[i+1] = x_true[i] + f(x_true[i]) * dt + L * dB

        # --- Compute expectations under q = N(mu, P) analytically for sin/cos ---
        # E[sin(x)] = exp(-P/2) * sin(mu)
        # E[cos(x)] = exp(-P/2) * cos(mu)
        E_sin = np.exp(-0.5 * P[i]) * np.sin(mu[i])
        E_cos = np.exp(-0.5 * P[i]) * np.cos(mu[i])

        # Cross-moment S = E[(x-mu) f(x)] = P * E[ f'(x) ] = P * E[cos(x)]
        S = P[i] * E_cos

        # KL-optimal linear params (scalar)
        A = S / P[i] if P[i] > 1e-12 else E_cos   # = E_cos (when P>0)
        b = E_sin - A * mu[i]
        Lambda = L

        # --- Linear SDE sample (using same dB) ---
        x_lin[i+1] = x_lin[i] + (A * x_lin[i] + b) * dt + Lambda * dB

        # --- Propagate variational mean & covariance (CD moment ODEs) ---
        # dmu/dt = E[sin(x)] = E_sin
        # dP/dt  = 2 * E[(x-mu) f(x)] + L^2 * Q = 2*S + L^2 * Q
        mu[i+1] = mu[i] + E_sin * dt
        P[i+1] = P[i] + (2.0 * S + (L**2) * Q) * dt

    true_trajs.append(x_true)
    lin_trajs.append(x_lin)
    mu_trajs.append(mu)
    P_trajs.append(P)

# Convert to arrays
true_trajs = np.array(true_trajs)
lin_trajs = np.array(lin_trajs)
mu_trajs = np.array(mu_trajs)
P_trajs = np.array(P_trajs)

# Plot all runs: true (blue), linear sim (green dashed), mean (orange)
plt.figure(figsize=(12,6))
for k in range(n_runs):
    plt.plot(time, true_trajs[k], color='blue', alpha=0.35, label='True SDE' if k == 0 else "")
    plt.plot(time, lin_trajs[k], color='green', alpha=0.35, linestyle='--', label='Linear SDE' if k == 0 else "")
    #plt.plot(time, mu_trajs[k], color='orange', alpha=0.35, linestyle='-.', label='CD mean' if k == 0 else "")

# Plot mean of mu across runs and ±2σ band using mean P across runs
mu_mean = np.mean(mu_trajs, axis=0)
P_mean = np.mean(P_trajs, axis=0)
#plt.plot(time, mu_mean, 'k-', lw=2, label='Mean over runs')
plt.fill_between(time, mu_mean - 2*np.sqrt(P_mean), mu_mean + 2*np.sqrt(P_mean),
                color='gray', alpha=0.25, label='±2σ (avg)')

plt.xlabel('Time', fontsize=18)
plt.ylabel('State x(t)', fontsize=18)
plt.title('5 runs: Nonlinear SDE (blue), Linear SDE surrogate (green dashed)', fontsize=18)
plt.legend(loc='upper left', fontsize=16)
plt.grid(True)
plt.show()
