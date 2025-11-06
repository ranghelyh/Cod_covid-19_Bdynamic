import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


Date = pd.to_datetime(df1['Day'])
I_data = df1['Daily new confirmed cases due to COVID-19 (rolling 7-day average, right-aligned)'].values
n_points = len(I_data)

# --- Parámetros fijos del modelo ---
beta_star = 0.3   # Intrinsic contagion rate of COVID-19
gamma = 1/10       # Removal rate of COVID-19
N = 19_660_000    # Population of Chile

# Condiciones iniciales
I0 = 16_340 # Number of people with COVID-19 10-07-2020
R0 = 10_753 # Number of removal people 10-07-2020
B0 = 0.2    # Estimated contagion rate of COVID-19 10-07-2020
S0 = N - I0 - R0 # Number of susceptible people 01-07-2020
y0 = [B0, S0, I0, R0]

t_span = (0, n_points - 1)
t_eval = np.linspace(*t_span, n_points)

# --- System β–SIR ---
def B_SIR(t, y, nu_max, I_nu_2, mu_max, I_mu_2):
    B, S, I, R = y
    # avoid division by zero
    I_pos = max(I, 1e-9)
    nu = nu_max * (I_nu_2 / (I_pos + I_nu_2))
    mu = mu_max * (I_pos / (I_pos + I_mu_2))
    dB_dt = +nu * (beta_star - B) - mu * B
    dS_dt = -B * S * I_pos / N
    dI_dt = +B * S * I_pos / N - gamma * I_pos
    dR_dt = +gamma * I_pos
    return [dB_dt, dS_dt, dI_dt, dR_dt]

# --- Function that returns the simulated series of new cases ---
def simulate_new_cases(params):
    # params = [nu_max, I_nu_2, mu_max, I_mu_2]
    nu_max, I_nu_2, mu_max, I_mu_2 = params

    sol = solve_ivp(
        lambda t, y: B_SIR(t, y, nu_max, I_nu_2, mu_max, I_mu_2),
        t_span, y0, t_eval=t_eval, method='RK45', vectorized=False, rtol=1e-6, atol=1e-8
    )
    if not sol.success:
    
        return np.full(n_points, 1e8)
    B_vals = sol.y[0]
    S_vals = sol.y[1]
    I_vals = sol.y[2]
    new_cases = B_vals * S_vals * I_vals / N
    
    if np.any(np.isnan(new_cases)) or np.any(new_cases < -1e-6):
        return np.full(n_points, 1e8)
 
    return new_cases


# --- Residuals para least_squares ---
def residuals_logspace(x):
    
    params = np.exp(x)
    sim = simulate_new_cases(params)
    return (sim - I_data)

# --- Initializations---
init = np.array([0.7, 2000.0, 0.8, 3800.0])  # nu_max, I_nu_2, mu_max, I_mu_2

x0 = np.log(init)

# limits on linear scale 
lb = [1e-4, 10.0, 1e-4, 10.0]   
ub = [5.0, 1e6, 5.0, 1e6]      

log_lb = np.log(lb)
log_ub = np.log(ub)

# --- Optimization ---
res = least_squares(
    residuals_logspace,
    x0,
    bounds=(log_lb, log_ub),
    verbose=2,
    xtol=1e-6,
    ftol=1e-6,
    max_nfev=200
)

# --- Adjusted parameters ---
fitted_params = np.exp(res.x)
nu_max_hat, I_nu_2_hat, mu_max_hat, I_mu_2_hat = fitted_params
print("Parámetros ajustados:")
print(f"nu_max = {nu_max_hat:.6f}")
print(f"I_nu_2 = {I_nu_2_hat:.3f}")
print(f"mu_max = {mu_max_hat:.6f}")
print(f"I_mu_2 = {I_mu_2_hat:.3f}")

# --- Simulate with adjusted parameters ---
sim_best = simulate_new_cases(fitted_params)

# --- Graphics ---
plt.figure(figsize=(12,5), dpi=300)
plt.plot(Date, sim_best, label='simulated',
    color='steelblue', linewidth= 1.5)
plt.scatter(Date, I_data, label='Observed', color='k', s=10, alpha=0.6)
plt.xlabel('Date', fontsize=16, fontweight='bold')
plt.ylabel('Daily cases', fontsize=16, fontweight='bold')
plt.xticks( fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/Modelo_SIR_Covid/simulation_vs_observed.png', dpi=600)
plt.show()
