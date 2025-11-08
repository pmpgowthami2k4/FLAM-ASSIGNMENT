import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# load data
data = pd.read_csv("xy_data.csv") 
x_true = data["x"].values
y_true = data["y"].values
n = len(x_true)

#model
def xy_from_t(t, theta, M, X):
    x = t * np.cos(theta) - np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.cos(theta)
    return x, y


def softplus(x):
    return np.log1p(np.exp(x))


def u_to_t(u, t_min=6.0, t_max=60.0):
    inc = softplus(u) + 1e-8
    cum = np.cumsum(inc)
    cum_min = cum[0]
    cum = cum - cum_min
    if cum[-1] == 0:
        scaled = np.zeros_like(cum)
    else:
        scaled = cum / cum[-1]
    t = t_min + scaled * (t_max - t_min)
    return t    

#Loss function

def loss_all(vars_vec, x_true, y_true, t_min=6.0, t_max=60.0, reg_t=1.0):
    theta = vars_vec[0]
    M = vars_vec[1]
    X = vars_vec[2]
    u = vars_vec[3:]
    t = u_to_t(u, t_min, t_max) #monotonic t
    x_pred, y_pred = xy_from_t(t, theta, M, X)
    l1 = np.sum(np.abs(x_true - x_pred) + np.abs(y_true - y_pred))
    t_lin = np.linspace(t_min, t_max, n) #regularization
    reg = reg_t * np.sum(np.abs(t - t_lin))
    return l1 + reg



# Initial guess
t_lin = np.linspace(6.0, 60.0, n)

A = np.vstack([t_lin, np.ones_like(t_lin)]).T
slope, intercept = np.linalg.lstsq(A, y_true - 42, rcond=None)[0]
slope = np.clip(slope, -0.999, 0.999)
theta0 = np.arcsin(slope)
M0 = 0.0
X0 = np.mean(x_true - t_lin * np.cos(theta0))

init_inc = np.ones(n)

u0 = np.log(np.exp(1.0) - 1.0) * np.ones(n)

vars0 = np.concatenate(([theta0, M0, X0], u0))


print("Initial theta (deg):", np.rad2deg(theta0), "M0:", M0, "X0:", X0)


#bounds
bnds = []
bnds.append((0.0, np.deg2rad(50.0))) #theta
bnds.append((-0.05, 0.05))            # M
bnds.append((0.0, 200.0))            # X    

large = 10.0

for i in range(n):
    bnds.append((-large, large))


# Optimization
res = minimize(loss_all, vars0, args=(x_true, y_true, 6.0, 60.0, 0.5), bounds=bnds, method='L-BFGS-B', options={'maxiter': 5000, 'ftol':1e-8})

print("Optimization success:", res.success, res.message)

theta_opt = res.x[0]
M_opt = res.x[1]
X_opt = res.x[2]
u_opt = res.x[3:]
t_opt = u_to_t(u_opt, 6.0, 60.0)
x_fit, y_fit = xy_from_t(t_opt, theta_opt, M_opt, X_opt)
final_L1 = np.sum(np.abs(x_true - x_fit) + np.abs(y_true - y_fit))
print("Final L1 + reg (objective):", res.fun)
print("Final pure L1:", final_L1)
print("theta_deg:", np.rad2deg(theta_opt), "M:", M_opt, "X:", X_opt)


df = pd.DataFrame({ 
    't_opt': t_opt,
    'x_true': x_true,
    'y_true': y_true,
    'x_fit': x_fit,
    'y_fit': y_fit,
    'abs_err_x': np.abs(x_true - x_fit),
    'abs_err_y': np.abs(y_true - y_fit)
})

df.to_csv("fit_with_t_optimization.csv", index=False)
pd.DataFrame([{'theta_rad':theta_opt, 'theta_deg':np.rad2deg(theta_opt), 'M':M_opt, 'X':X_opt, 'L1':final_L1}]).to_csv('fit_params_with_t.csv', index=False)


plt.figure(figsize=(8,5))
plt.scatter(x_true, y_true, label='Data', s=30)
plt.plot(x_fit, y_fit, '-r', linewidth=2, label='Fitted (optimized t)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

plt.title('Fit with per-point t optimization (monotonic enforced)')
plt.show()

# LaTeX-ready equations
print("\nLaTeX-ready equations:")
print("theta (rad) =", theta_opt, "theta (deg) =", np.rad2deg(theta_opt))
print("M =", M_opt, "X =", X_opt)
print("Note: t_i (per-point) are in 'fit_with_t_optimization.csv' - use those t to evaluate x(t), y(t).")
