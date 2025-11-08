# FLAM-ASSIGNMENT

## Problem Statement:

To find the values of unknown variables in the given **parametric equation of a curve**:

$$
x = \left( t \cdot \cos(\theta) - e^{M|t|} \cdot \sin(0.3t) \cdot \sin(\theta) + X \right)
$$

$$
y = \left( 42 + t \cdot \sin(\theta) + e^{M|t|} \cdot \sin(0.3t) \cdot \cos(\theta) \right)
$$


where \( t \) varies within the range \( 6 < t < 60 \).

---

###  Unknown Parameters

The parameters **θ**, **M**, and **X** are unknown and must be estimated.

**Given parameter bounds:**
- \( 0° < \theta < 50° \)
- \( -0.05 < M < 0.05 \)
- \( 0 < X < 100 \)

## Our understanding of the problem

The problem is a parameter estimation task for a nonlinear parametric curve. 

We have several known data points that lie on a curve described by mathematical equations, but we don’t know the exact values of some constants (θ, M, and X) that define the curve’s shape and position.

**The goal is to find these missing parameters so that the generated curve aligns as closely as possible with the real data.**

The chosen metric for comparison is the L1 distance, which measures the total absolute difference between the model’s predictions and the experimental data, making the optimization robust against outliers.



## My approach:

The central idea is to use computational techniques to minimize the difference between the model-generated curve and the measured data points by tuning the parameters automatically.

The unknown parameters were estimated by formulating the problem as an optimization task.  
The program calculates both the predicted and actual values of the coordinates (x, y) and measures their deviation using the L1 distance, defined as:

$$
L_{1} = \sum_{i} \left( \left| x_{i}^{\text{true}} - x_{i}^{\text{pred}} \right| + \left| y_{i}^{\text{true}} - y_{i}^{\text{pred}} \right| \right)
$$


The objective of the optimization is to minimize this L1 value, which directly corresponds to how well the predicted curve fits the given data.



## How does it work?

### 1. Data Loading

```python
data = pd.read_csv("xy_data.csv")
x_true = data["x"].values
y_true = data["y"].values
n = len(x_true)
```

### 2. Defining the Curve

```python
def xy_from_t(t, theta, M, X):
    x = t * np.cos(theta) - np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.cos(theta)
    return x, y
```

### 3. t-value mapping

Since t values are unknown, internal variables ‘u’ are introduced.
These are transformed using the softplus function to generate positive increments that are then normalized to produce increasing t values.

```python
def softplus(x):
    return np.log1p(np.exp(x))

def u_to_t(u, t_min=6.0, t_max=60.0):
    # u: shape (n,) raw unconstrained variables
    # convert to positive increments
    inc = softplus(u) + 1e-8  # ensure strictly positive
    cum = np.cumsum(inc)      # strictly increasing
    # normalize to [t_min, t_max]
    cum_min = cum[0]
    cum = cum - cum_min  # start from 0
    if cum[-1] == 0:
        scaled = np.zeros_like(cum)
    else:
        scaled = cum / cum[-1]    # in [0,1]
    t = t_min + scaled * (t_max - t_min)
    return t
```

### 4. Loss function Definition

The total L1 distance between predicted and true data is computed,
with a small regularization term to prevent overfitting by encouraging evenly spaced t values.

```python
def loss_all(vars_vec, x_true, y_true, t_min=6.0, t_max=60.0, reg_t=1.0):
    theta = vars_vec[0]
    M = vars_vec[1]
    X = vars_vec[2]
    u = vars_vec[3:]
    t = u_to_t(u, t_min, t_max)
    x_pred, y_pred = xy_from_t(t, theta, M, X)
    l1 = np.sum(np.abs(x_true - x_pred) + np.abs(y_true - y_pred))
    t_lin = np.linspace(t_min, t_max, n)
    reg = reg_t * np.sum(np.abs(t - t_lin))
    return l1 + reg
```

### 5. Setting Parameters and bounds

Before optimization, the code calculates:

* 𝜃 is estimated from the slope of a linear regression between t and y
* M starts at zero (neutral exponential factor)
* X is set based on the mean horizontal offset

```python
t_lin = np.linspace(6.0, 60.0, n)
A = np.vstack([t_lin, np.ones_like(t_lin)]).T
slope, intercept = np.linalg.lstsq(A, (y_true - 42), rcond=None)[0]
slope = np.clip(slope, -0.999, 0.999)
theta0 = np.arcsin(slope)
M0 = 0.0
X0 = np.mean(x_true - t_lin * np.cos(theta0))

bnds = []
bnds.append((0.0, np.deg2rad(50.0)))  # theta
bnds.append((-0.05, 0.05))            # M
bnds.append((0.0, 200.0))             # X
for i in range(n):
    bnds.append((-10.0, 10.0))        # u_i unbounded
```

### 6. Optimization

The `minimize()` function executes the optimization using the **L-BFGS-B** algorithm,
which efficiently handles large problems with simple bounds.
It iteratively updates 𝜃, M, X, t to minimize the loss function.


## Mathematical Proof:

| | |
|:--:|:--:|
| ![Proof 1](https://github.com/user-attachments/assets/7a2ebace-5688-4278-a457-d71100d9c7f3) | ![Proof 2](https://github.com/user-attachments/assets/8223027c-a028-4015-9aea-fd221d2f5240) |

## Output:
<img width="977" height="165" alt="image" src="https://github.com/user-attachments/assets/46c1b550-a8fb-4dbd-ace8-30503441729d" />

### Output Variables:

| Variable                   | Value       |                            |
| -------------------------- | ----------- | --------------------------------------- |
| initial θ (deg)            | **19.84**   | Initial guess for the angle θ           |
| M0                         | **0.0**     | Initial guess for exponential parameter |
| X0                         | **85.27**   | Initial guess for horizontal offset     |
| Optimize success           | **True**    | Optimization status                     |
| Final Objective (L1 + reg) | **47.24**   | Total minimized cost                    |
| Final L1                   | **46.85**   | Pure fitting error                      |
| θ                          | **19.63°**  | Optimized angle parameter               |
| M                          | **-0.0014** | Optimized exponential distortion        |
| X                          | **83.72**   | Optimized X-offset                      |

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/07a2b4b4-636d-4605-a217-1f9c25cb6026" />


## Plot Interpretation

The plot displayed after the optimization shows two sets of points:

* 🔵 **Blue Scatter Points** → Original data points from the dataset
* 🔴 **Red Line (Smooth Curve)** → Fitted curve generated using the optimized parameters



## Interpretation of the Plot

### Curve Alignment

The red fitted curve passes closely through the blue data points, indicating a low L1 error and successful fit.

### Smoothness

The curve is continuous and monotonic in the *t*-direction, confirming that the softplus mapping worked correctly.

### Shape Behavior

The exponential factor **M = -0.0014** slightly flattens oscillations, making the curve smoother and more stable.

### Angle θ Effect

The fitted **θ = 19.63°** gives the curve a tilted orientation consistent with the data trend.

### Offset X Effect

The value **X = 83.72** shifts the entire curve horizontally to match the center of the data.






## Why this approach?

The **L1 optimization method** was chosen because it is the only approach that satisfies all the **mathematical, numerical, and practical requirements** of the given problem in a single framework.

The assignment demanded **minimizing the L1 distance** between measured and predicted data while keeping the parameters within **strict physical limits** — something that basic fitting or manual tuning methods cannot reliably achieve.

By using the **L-BFGS-B algorithm**, the model could **optimize parameters under defined bounds**.
The **softplus-based time mapping** effectively solved the missing time variable problem.
Adding a **regularization term** prevented overfitting and improved model stability.







