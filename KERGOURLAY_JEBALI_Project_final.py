import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt

##################
### Question 1 ###
##################

# problem dimensions
n = 20
m = 30
# set random seed for reproducibility
np.random.seed(1) 
# generate random data
A = np.random.randn(m, n)
b = np.random.randn(m)

# define the optimization problem
x = cp.Variable(n) 
objective = cp.Minimize(cp.norm(A @ x - b, 2))
constraints = [x >= 0,  x <= 1]

# create problem and solve using CVXPY
prob = cp.Problem(objective, constraints)
prob.solve()

# print results
print("\n\nStatus:", prob.status)             
print(f"Optimal value: {prob.value:.6f}")  # optimal value of the objective function
print(f"Optimal x: {x.value} \n\n")          # optimal solution vector


##################
### Question 2 ###
##################

n_values = [20,200]           # list of n values tested
m_values = [30,300]           # list of m values tested
computation_times = []    # emplty list to store computation times

for n, m in zip(n_values, m_values): 
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(A @ x - b, 2))
    constraints = [x >= 0, x <= 1]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    computation_times.append(prob.solver_stats.solve_time) # solver_stats.solve_time gives computation time

# plotting computation times as a function of of n and m as they grow
plt.figure()
plt.plot(n_values, computation_times, marker='o')
plt.xlabel('Problem Size (n=m/1.5)')
plt.ylabel('Computation Time (seconds)')
plt.title('Computation Time vs Problem Size (CVXPY)')
plt.grid()


##################
### Question 3 ###
##################

from scipy.optimize import minimize
import time

n_values = [20, 100, 500] 
m_values = [30, 150, 750]

def f(x, A, b):       # objective function
    return np.linalg.norm(A @ x - b, 2)

def constraint0(x):   # constraint function for x >= 0
    return x

def constraint1(x):   # constraint function for x <= 1
    return 1 - x

computation_times_cvxpy = [] 
computation_times_scipy = []

for n, m in zip(n_values, m_values):
    print(f"Testing for n={n}, m={m}...")
    
    np.random.seed(1) 
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    x_cvx = cp.Variable(n)
    objective = cp.Minimize(cp.norm(A @ x_cvx - b, 2))
    constraints_cvx = [x_cvx >= 0, x_cvx <= 1] # Ajout contrainte <= 1 pour être identique à Scipy
    prob = cp.Problem(objective, constraints_cvx)
    
    start_time = time.time()
    prob.solve()
    end_time = time.time()
    computation_times_cvxpy.append(prob.solver_stats.solve_time)
    solution_cvx = x_cvx.value

    x0 = np.zeros(n)
    constraints_scipy = [{'type': 'ineq', 'fun': constraint0}, 
                         {'type': 'ineq', 'fun': constraint1}]

    start_time = time.time()
    res = minimize(f, x0, args=(A, b), constraints=constraints_scipy, method='SLSQP')
    end_time = time.time()
    
    computation_times_scipy.append(end_time - start_time)
    solution_scipy = res.x

    diff = np.linalg.norm(solution_cvx - solution_scipy)
    print(f"  Difference norm solutions: {diff:.2e}")
    print(f"  Time CVXPY: {computation_times_cvxpy[-1]:.4f}s")
    print(f"  Time SCIPY: {computation_times_scipy[-1]:.4f}s\n")

plt.figure()
plt.plot(n_values, computation_times_scipy, marker='o', color='orange')
plt.xlabel('Problem Size (n)')
plt.ylabel('Computation Time (seconds)')
plt.title('Computation Time vs Problem Size (SCIPY)')
plt.grid()

plt.figure()
plt.plot(n_values, computation_times_cvxpy, marker='o', label='CVXPY')
plt.plot(n_values, computation_times_scipy, marker='o', label='SCIPY', color='orange')
plt.xlabel('Problem Size (n)')
plt.ylabel('Computation Time (seconds)')
plt.title('Computation Time Comparison (CVXPY vs SCIPY)')
plt.legend()
plt.grid()


##################
### Question 4 ###
##################

def markovitz_portfolio(mu, sigma):
    n = mu.shape[1] # dimension of the problem
    rmin = 1.6/252  # given value of the minimum return constraint

    x = cp.Variable((n,1))                          # optimization variable (we put (n,1) to force it to be a column vector as wanted by compute_metrics) 
    objective = cp.Minimize(cp.quad_form(x, sigma)) # objective function (portfolio variance)
    constraints = [                                 # constraints
        cp.sum(x) == 1,          # fully invested, use of total budget
        mu @ x >= rmin,          # minimum return constraint
        x >= 0                   # no short positions
    ]

    # create and solve the problem with CVXPY
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value


##################
### Question 5 ###
##################

from supporting_functions import download_finance_data, compute_moments, compute_metrics

# we load the training data and test data with 3 assets
Ytrain, Ytest = download_finance_data(3)
mu, sigma = compute_moments(Ytrain)

sigma = (sigma + sigma.T) / 2

x = markovitz_portfolio(mu, sigma)

# we use the function compute_metrics(x, Ytrain, Ytest) to understand the expected return 
# and its variance on the training and test data.
s_tr, s_te = compute_metrics(x, Ytrain, Ytest)

# now, we change the number of assets from 3 to 20.
Ytrain, Ytest = download_finance_data(20)
mu, sigma = compute_moments(Ytrain)
x = markovitz_portfolio(mu, sigma)
s_tr, s_te = compute_metrics(x, Ytrain, Ytest)


##################
### Question 6 ###
##################

from supporting_functions import download_finance_data, compute_moments, compute_metrics

# now we shift to the probabilistic variant of the portfolio optimization problem.

import scipy

def markovitz_portfolio_probabilistic(mu, sigma):
    n = mu.shape[1]          # dimension of the problem
    alpha = 1.6 / 252        # given value of the minimum return constraint
    beta = 0.49              # given probability level
    invphi_beta = scipy.stats.norm.ppf(beta)  # inverse CDF of standard normal at beta

    x = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(x, sigma))
    sigma_sqrt = scipy.linalg.sqrtm(sigma)
    constraints = [ 
        cp.sum(x) == 1,                                         # fully invested, use of total budget
    #   mu @ x + invphi_beta*cp.quad_form(x, sigma) >= alpha,   # probabilistic return constraint # quad_form(x, sigma) is variance = 
        x >= 0,                                                 # no short positions
        mu @ x + invphi_beta*cp.norm(sigma_sqrt @ x, 2) >= alpha
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve() 
    x_df = pd.DataFrame(x.value.reshape(-1,1))
    return x_df

# now we analyse the results with the training and testing data and compare with the standard version.
Ytrain, Ytest = download_finance_data(3)
mu, sigma = compute_moments(Ytrain)
x_prob = markovitz_portfolio_probabilistic(mu, sigma)

x_prob = pd.DataFrame(
    x_prob.values.reshape(-1,1),
    index=Ytrain.columns,
    columns=["w"]
)

s_tr_prob, s_te_prob = compute_metrics(x_prob, Ytrain, Ytest)

print("\nProbabilistic Portfolio with 3 assets:")
print("Training set:\n", s_tr_prob, "\n")
print("Testing set:\n", s_te_prob, "\n")

Ytrain, Ytest = download_finance_data(20)
mu, sigma = compute_moments(Ytrain)
x_prob = markovitz_portfolio_probabilistic(mu, sigma)

x_prob = pd.DataFrame(
    x_prob.values.reshape(-1,1),
    index=Ytrain.columns,
    columns=["w"]
)

s_tr_prob, s_te_prob = compute_metrics(x_prob, Ytrain, Ytest)

print("\nProbabilistic Portfolio with 20 assets:")
print("Training set:\n", s_tr_prob, "\n")
print("Testing set:\n", s_te_prob, "\n")


##################
### Question 7 ###
##################

# The testing set is limited to 2020, which was an odd year due to COVID.
# so we modify the download_finance_data function to include 2021 in the testing set.

def download_finance_data2(n_assets=10):

    # Tickers of assets
    assets = ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'MMC', 'JPM',
              'ZION', 'PSA', 'BAX', 'BMY', 'LUV', 'PCAR', 'TXT', 'TMO',
                'MSFT', 'HPQ', 'SEE', 'VZ', 'CNP', 'NI', 'T', 'BA']
    assets.sort()

    # Downloading data
    if n_assets>23:
        print('Warning: max number of assets is limited to 23')
        n_assets = 23

    # Training
    training_data = yf.download(assets[:n_assets], start='2016-01-01', end='2019-12-30', group_by="ticker")

    # Testing
    testing_data = yf.download(assets[:n_assets], start='2020-01-01', end='2021-12-30', group_by="ticker")

    Y = dict()
    # Compute the monthly returns:
    for ast in assets[:n_assets]:
        qq = training_data[ast]['Close']
        Y[ast] = [100*(qq[ii]- qq[ii-1])/qq[ii-1] for ii in range(1,len(qq))]

    training_df = pd.DataFrame(data=Y)

    Y = dict()
    # Compute the monthly returns:
    for ast in assets[:n_assets]:
        qq = testing_data[ast]['Close']
        Y[ast] = [100 * (qq[ii] - qq[ii - 1]) / qq[ii - 1] for ii in range(1, len(qq))]

    testing_df = pd.DataFrame(data=Y)

    return training_df, testing_df

Ytrain, Ytest = download_finance_data2(3)
mu, sigma = compute_moments(Ytrain)
x = markovitz_portfolio_probabilistic(mu, sigma)

x = pd.DataFrame(
    x.values.reshape(-1,1),
    index=Ytrain.columns,
    columns=["w"]
)

s_tr, s_te = compute_metrics(x, Ytrain, Ytest)
print("\nPortfolio with 3 assets (2020-2021 testing):")
print("Training set:\n", s_tr, "\n")
print("Testing set:\n", s_te, "\n")

Ytrain, Ytest = download_finance_data2(20)
mu, sigma = compute_moments(Ytrain)
x = markovitz_portfolio_probabilistic(mu, sigma)

x = pd.DataFrame(
    x.values.reshape(-1,1),
    index=Ytrain.columns,
    columns=["w"]
)

s_tr, s_te = compute_metrics(x, Ytrain, Ytest)
print("\nPortfolio with 20 assets (2020-2021 testing):")
print("Training set:\n", s_tr, "\n")
print("Testing set:\n", s_te, "\n")


##################
### Question 8 ###
##################

# let's plot the time-series data of the assets (use 3 assets for clarity) in the training and testing phase. 

Ytrain, Ytest = download_finance_data2(3)

plt.figure(figsize=(12, 6))

for asset in Ytrain.columns: # loop over each asset
    plt.plot(Ytrain.index, Ytrain[asset], label=f'Training {asset}') # Ytrain.index gives the dates, Ytrain[asset] gives returns 

for asset in Ytest.columns:  # loop over each asset
    plt.plot(Ytest.index, Ytest[asset], label=f'Testing {asset}', linestyle='--')

plt.xlabel('Date')
plt.ylabel('Returns (%)')
plt.title('Asset Returns Over Time')
plt.legend()
plt.grid()


##################
### Question 9 ###
##################

from supporting_functions import montecarlo_sim, scatter_plot_port

Ytrain, Ytest = download_finance_data2(3)

n_samples = 1000
n_assets = 3
c_mean, c_var = montecarlo_sim(n_assets, n_samples, Ytrain)
scatter_plot_port(c_mean, c_var, s_tr.iloc[0], s_tr.iloc[1])
c_mean, c_var = montecarlo_sim(n_assets, n_samples, Ytest)
scatter_plot_port(c_mean, c_var, s_te.iloc[0], s_te.iloc[1])

# now we change the number of assets to 20.

Ytrain, Ytest = download_finance_data2(20)

n_samples = 1000
n_assets = 20
c_mean, c_var = montecarlo_sim(n_assets, n_samples, Ytrain)
scatter_plot_port(c_mean, c_var, s_tr.iloc[0], s_tr.iloc[1])
c_mean, c_var = montecarlo_sim(n_assets, n_samples, Ytest)
scatter_plot_port(c_mean, c_var, s_te.iloc[0], s_te.iloc[1])


##################
## Question 10 ###
##################

# The problem (5) is nonconvex because of the following constraint X=x*x.T which is not convex. 


###################
### Question 11 ###
###################

n = 20
m = 30
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# let's code a SQP solver for the given problem. 

def sqp_solver(A, b, x0, max_iter=50, tol=1e-4):
    
    # objective function f(x) = 1/2 ||Ax - b||^2
    def f(x):
        return 0.5 * np.linalg.norm(A @ x - b, 2)**2

    # gradient : ∇f(x) = A^T (Ax - b)
    def grad_f(x):
        return A.T @ (A @ x - b)
    
    # hessian : ∇²f(x) = A^T A
    def hess_f(x):
        return A.T @ A
    
    # constraints g(x) <= 0 (2n+1 constraints)
    def g(x):
        g_val = np.hstack([
            -x,                     # x >= 0
            x - 1,                  # x <= 1
            [0.5 - np.sum(x**2)]    # ||x||^2 >= 0.5
        ])
        return g_val

    # jacobian of constraints ∇g(x) (shape (2n+1, n))     
    def jacobian_g(x):
        n_vars = x.shape[0]
        # for -x <= 0  -> -Identity
        J1 = -np.eye(n_vars)
        # for x - 1 <= 0 -> Identity
        J2 = np.eye(n_vars)
        # for 0.5 - ||x||^2 -> -2x
        J3 = -2 * x.reshape(1, -1)
        
        return np.vstack([J1, J2, J3])
    
    x_k = x0.copy()
    step_norms = []    # to track the norm of steps ||dx|| to monitor convergence
    cost_history = []  # to store objective function values
    
    H = hess_f(x_k)    # constant hessian for this problem
    H = (H + H.T) / 2  # ensure symmetry
    
    print(f"\n\nStart of SQP Solver (Max iter: {max_iter})...")

    for k in range(max_iter):
        
        grad_k = grad_f(x_k)
        g_k = g(x_k)
        jac_k = jacobian_g(x_k)
        
        delta_x = cp.Variable(n)
        cost_approx = (grad_k @ delta_x) + 0.5 * cp.quad_form(delta_x, H)
        constraints = [g_k + jac_k @ delta_x <= 0]
        
        prob = cp.Problem(cp.Minimize(cost_approx), constraints)
        
        prob.solve(solver=cp.OSQP, verbose=False) 
        
        if delta_x.value is None:
            print(f"\n QP Solver failed at iteration {k}")
            break
            
        dx = delta_x.value
        step_norm = np.linalg.norm(dx)
        
        x_k = x_k + dx
        
        step_norms.append(step_norm)
        cost_history.append(f(x_k))
        
        # stop criteria: if the step is small enough then stop
        if step_norm < tol:
            print(f"\n Convergence reached at iteration {k}. Step norm: {step_norm:.2e}")
            break

    return x_k, step_norms, cost_history

# random initialization (important to not start at 0 to avoid zero gradient)
x_init = np.random.rand(n)
x_opt, steps, costs = sqp_solver(A, b, x_init)

print("\n--- Optimal Solution ---")
print(f"Optimal x:\n{x_opt}\n")
print(f"Final Objective Value: {costs[-1]:.4f}\n")

# let's plot the norm of the residual and the objective function value over iterations
plt.figure()
plt.plot(steps, 'o-')
plt.title('Evolution of step size ||dx||')
plt.ylabel('||dx||')
plt.xlabel('Iteration')
plt.grid()

plt.figure()
plt.plot(costs, 'r-')
plt.title('Evolution of objective function value')
plt.ylabel('f(x)')
plt.xlabel('Iteration')
plt.grid()


###################
### Question 12 ###
###################

import scipy.optimize as sp
import time

def func_obj(x):
    return 0.5 * np.linalg.norm(A @ x - b)**2

def func_grad(x):
    return A.T @ (A @ x - b)

def constraint_norm(x):
    return np.sum(x**2) - 0.5

cons = [
    {'type': 'ineq', 'fun': constraint_norm} # inequality means fun(x) >= 0
]
# bounds 0 <= x <= 1
bnds = tuple((0, 1) for _ in range(n))

print("\n--- SCIPY SLSQP Execution ---")
start_time = time.time()

res_scipy = sp.minimize(
    fun=func_obj,
    x0=x_init,           # same initial point as Q11
    jac=func_grad,       # we provide the analytic gradient
    method='SLSQP',
    bounds=bnds,
    constraints=cons,
    options={'disp': True, 'maxiter': 100}
)

end_time = time.time()
scipy_time = end_time - start_time

print("-" * 40)
print(f"Custom SQP Final Objective: {costs[-1]:.6f}")
print(f"Scipy SLSQP Final Objective: {res_scipy.fun:.6f}")
print(f"Scipy Time: {scipy_time:.4f} seconds")
print(f"Scipy Success: {res_scipy.success}")
print(f"Scipy Iterations: {res_scipy.nit}")

diff_norm = np.linalg.norm(x_opt - res_scipy.x)
print(f"Difference between solutions ||x_custom - x_scipy||: {diff_norm:.6e}")


###################
### Question 13 ###
###################

# let's solve the portfolio Kurtosis Minimization Problem with SCP using perturbation variables

from supporting_functions import compute_moments, compute_coefficients

n_assets = 20
Ytrain, Ytest = download_finance_data2(n_assets)
mu, sigma = compute_moments(Ytrain)
r_min = 1.6 / 252  

D2, L2, S4 = compute_coefficients(Ytrain, n_assets)
S2 = D2.T 
M_matrix = S2 @ S4 @ S2.T

# for numerical Stability,  we use eigenvalue decomposition to compute the square root and we force eigenvalues to be non-negative.
eigvals, eigvecs = np.linalg.eigh(M_matrix)
eigvals[eigvals < 0] = 0
M_soc = np.real(eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T)

# SCP is a local optimization method; it needs a feasible starting point.
# We solve the standard Minimum Variance problem (Convex QP) to get x0.
x_var = cp.Variable(n_assets)
prob_init = cp.Problem(
    cp.Minimize(cp.quad_form(x_var, sigma)), 
    [cp.sum(x_var) == 1, x_var >= 0, mu @ x_var >= r_min]
)
prob_init.solve(solver=cp.ECOS)

# Retrieve value or fallback to equal-weight if solver fails
x_k = x_var.value
if x_k is None: 
    x_k = np.ones(n_assets)/n_assets 

x_k = x_k.flatten() # 1D numpy array to avoid dimension broadcasting errors

X_k = np.outer(x_k, x_k) # initialize the "lifted" variables for the first iteration: X_k = x_k * x_k.T (Rank-1 Matrix)

z_k = L2 @ X_k.flatten(order='F') # we must use order='F' (Fortran/Column-major) because L2 was likely generated 

g_k = np.linalg.norm(M_soc @ z_k)

# print initial status using .item() to extract scalar from numpy array (fixes format error)
rendement_init = (mu @ x_k).item()
print(f"\nInitial State: Return={rendement_init:.5f}, Kurtosis param (g)={g_k:.5f}")

max_iter = 20
tolerance = 1e-4      # convergence tolerance for step size
g_history = []        # to store objective value history
step_history = []     # to store step size norm history

print("\n")
print("STARTING SCP ALGORITHM (Perturbation Method)")
print("\n")

for k in range(max_iter):
    
    # we define perturbation variables (deltas) and solve the convex sub-problem at each iteration.
    dx = cp.Variable(n_assets)
    dX = cp.Variable((n_assets, n_assets), symmetric=True)
    dz = cp.Variable(len(z_k))
    dg = cp.Variable()
    
    constraints = []
    constraints.append(cp.sum(x_k + dx) == 1)      # Budget constraint
    constraints.append(x_k + dx >= 0)              # No short selling
    constraints.append(mu @ (x_k + dx) >= r_min)   # Minimum Return
    
    # linearization of the non-convex rank-1 constraint: X = xx^T ; we use First-order Taylor Expansion around x_k    
    term1 = cp.reshape(x_k, (n_assets, 1), order='F') @ cp.reshape(dx, (1, n_assets), order='F') # cp.reshape to ensure (N,1) @ (1,N) multiplication
    term2 = cp.reshape(dx, (n_assets, 1), order='F') @ cp.reshape(x_k, (1, n_assets), order='F')
    
    constraints.append( X_k + dX == np.outer(x_k, x_k) + term1 + term2 ) 
    constraints.append( z_k + dz == L2 @ cp.vec(X_k + dX, order='F') )
    constraints.append( cp.SOC(g_k + dg, M_soc @ (z_k + dz)) )
    constraints.append( cp.norm(dx) <= 0.1 )
    
    prob = cp.Problem(cp.Minimize(g_k + dg), constraints)
    prob.solve(solver=cp.ECOS, verbose=False)
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Iter {k}: Solver failed with status ({prob.status})")
        break
        
    dx_val = dx.value
    step_norm = np.linalg.norm(dx_val)
    
    x_k += dx_val
    X_k += dX.value
    z_k += dz.value
    g_k += dg.value
    
    g_history.append(g_k)
    step_history.append(step_norm)
    
    print(f"Iter {k+1}: ||dx|| = {step_norm:.6f}, Objective g = {g_k:.6f}")
    
    if step_norm < tolerance: # check convergence
        print("Convergence reached.")
        break


plt.figure()
plt.plot(step_history, 'o-')
plt.title('Convergence of Step Size (||dx||)')
plt.xlabel('Iteration')
plt.ylabel('Norm of perturbation')
plt.grid(True)

plt.figure()
plt.plot(g_history, 's-', color='orange')
plt.title('Minimization of Kurtosis Parameter (g)')
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.grid(True)

plt.show()

# print final portfolio composition
df_res = pd.DataFrame(x_k, index=Ytrain.columns, columns=['Weight'])
print("\nTop 5 Assets in Min-Kurtosis Portfolio:")
print(df_res.sort_values(by='Weight', ascending=False).head(5))
print("\n\n")


###################
### Question 14 ###
###################

# let's solve the Portfolio Kurtosis Problem (5) using SCIPY.optimize and the SLSQP solver

import scipy.optimize as sp
import time

def kurtosis_objective(x, L2, M_soc):
    # lift variables: X = x * x.T
    X = np.outer(x, x)
    
    # vectorize X to z (Using Fortran order to match L2 construction)
    z = L2 @ X.flatten(order='F')
    
    g = np.linalg.norm(M_soc @ z)
    return g

constraints_slsqp = [
    {'type': 'eq',  'fun': lambda x: np.sum(x) - 1},
    {'type': 'ineq', 'fun': lambda x: mu @ x - r_min}
]

bounds_slsqp = tuple((0, 1) for _ in range(n_assets))

print("STARTING SCIPY SLSQP OPTIMIZATION")
print("-" * 40)

# we use the same starting point x0 (Min Variance) calculated in Q13 to ensure a fair comparison, as SLSQP is a local solver.
x0_slsqp = x_var.value.flatten() # Ensure 1D array

start_time = time.time()

result_slsqp = sp.minimize(
    fun=kurtosis_objective,
    x0=x0_slsqp,
    args=(L2, M_soc),      # Pass the structural matrices as arguments
    method='SLSQP',
    bounds=bounds_slsqp,
    constraints=constraints_slsqp,
    options={'disp': True, 'ftol': 1e-6, 'maxiter': 100}
)

end_time = time.time()
computation_time = end_time - start_time

# compare and print results
print("-" * 40)
if result_slsqp.success:
    print("SLSQP Optimization Successful")
else:
    print(f"SLSQP Failed: {result_slsqp.message}")

x_opt_slsqp = result_slsqp.x
final_obj_slsqp = result_slsqp.fun

print(f"Computation Time: {computation_time:.4f} seconds")
print(f"Final Kurtosis Objective (g): {final_obj_slsqp:.6f}")

print(f"SCP Objective (Q13):          {g_k:.6f}")
print(f"Difference (SLSQP - SCP):     {final_obj_slsqp - g_k:.6e}")

df_res_slsqp = pd.DataFrame(x_opt_slsqp, index=Ytrain.columns, columns=['Weight_SLSQP'])
print("\nTop 5 Assets in SLSQP Portfolio:")
print(df_res_slsqp.sort_values(by='Weight_SLSQP', ascending=False).head(5))

print("\nConstraint Check:")
print(f"Sum of weights: {np.sum(x_opt_slsqp):.6f} (Target: 1.0)")
print(f"Portfolio Return: {(mu @ x_opt_slsqp).item():.6f} (Target: >= {r_min:.6f})")
print("\n")


###################
### Question 15 ###
###################

from supporting_functions import compute_moments, compute_metrics, montecarlo_sim

# Minimum Variance Portfolio (Benchmark)
x_mv = cp.Variable(n_assets)
prob_mv = cp.Problem(cp.Minimize(cp.quad_form(x_mv, sigma)), 
                     [cp.sum(x_mv) == 1, x_mv >= 0, mu @ x_mv >= r_min])
prob_mv.solve(solver=cp.ECOS)
w_mv = x_mv.value.flatten()

# Minimum Kurtosis Portfolio (Result from Q13/Q14)
# we use x_k (or x_opt_slsqp) calculated previously
w_kurt = x_k.flatten()

df_mv = pd.DataFrame(w_mv, index=Ytrain.columns, columns=[0])
df_kurt = pd.DataFrame(w_kurt, index=Ytrain.columns, columns=[0])

_, res_mv = compute_metrics(df_mv, Ytrain, Ytest)
_, res_kurt = compute_metrics(df_kurt, Ytrain, Ytest)

# we extract scalar values for plotting
ret_mv = res_mv.iloc[0].item()
std_mv = res_mv.iloc[1].item() 
ret_kurt = res_kurt.iloc[0].item()
std_kurt = res_kurt.iloc[1].item()

print(f"MinVar (Test): Return={ret_mv:.4f}, Volatility={std_mv:.4f}")
print(f"MinKurt (Test): Return={ret_kurt:.4f}, Volatility={std_kurt:.4f}")

# monte carlo & scatter plot
n_samples = 1000 
n_assets_mc = 20

print("\n--- Generating Monte Carlo (Test Data) ---")
c_mean, c_var = montecarlo_sim(n_assets_mc, n_samples, Ytest)
c_std = np.sqrt(c_var) 

plt.figure(figsize=(10, 6))

# cloud of random portfolios
plt.scatter(c_std, c_mean, c=c_mean/c_std, cmap='viridis', marker='o', s=15, alpha=0.5, label='Random')
plt.colorbar(label='Sharpe Ratio')

# Minimum Variance Portfolio (Blue Square)
plt.scatter(std_mv, ret_mv, color='blue', marker='s', s=150, label='Min Variance', zorder=5)

# Minimum Kurtosis Portfolio (Red Star)
plt.scatter(std_kurt, ret_kurt, color='red', marker='*', s=250, label='Min Kurtosis (SCP)', zorder=5)

plt.title("Risk-Return Comparison: Min Variance vs Min Kurtosis (Test Data 2020-2021)")
plt.xlabel("Volatility (Standard Deviation)")
plt.ylabel("Expected Return")
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# pie charts for comparison
def plot_pie_manual(weights, title, assets):
    mask = weights > 0.01 
    w_main = weights[mask]
    labels = assets[mask]
    
    plt.figure(figsize=(6, 6))
    plt.pie(w_main, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.show()

print("Generating Pie Charts...")
plot_pie_manual(w_mv, "Composition: Min Variance", Ytrain.columns)
plot_pie_manual(w_kurt, "Composition: Min Kurtosis (SCP)", Ytrain.columns)


###################
### Question 17 ###
###################

from scipy.linalg import sqrtm

print("\n" + "="*50)
print("STARTING CONVEX RELAXATION (SDP)")
print("="*50)

x_sdp = cp.Variable((n_assets, 1)) 
X_sdp = cp.Variable((n_assets, n_assets), symmetric=True)
z_sdp = cp.Variable((z_k.shape[0], 1))
g_sdp = cp.Variable()

constraints_sdp = []
constraints_sdp.append(cp.sum(x_sdp) == 1)
constraints_sdp.append(x_sdp >= 0)
constraints_sdp.append(mu @ x_sdp >= r_min)
constraints_sdp.append( z_sdp == L2 @ cp.vec(X_sdp, order='F') )
constraints_sdp.append( cp.SOC(g_sdp, M_soc @ z_sdp) )

# SDP Relaxation (Schur Complement)
# we use cp.bmat to construct the block matrix
block_matrix = cp.bmat([
    [X_sdp, x_sdp],
    [x_sdp.T, np.array([[1.0]])]
])
constraints_sdp.append( block_matrix >> 0 ) # ">> 0" means Positive Semi-Definite

prob_sdp = cp.Problem(cp.Minimize(g_sdp), constraints_sdp)

start_time = time.time()
prob_sdp.solve(verbose=False)
end_time = time.time()
sdp_time = end_time - start_time

if prob_sdp.status in ["optimal", "optimal_inaccurate"]:
    print(f"SDP Solved Successfully ({prob_sdp.status})")
    print(f"Time: {sdp_time:.4f} seconds")
    print(f"Optimal Relaxed Objective (g): {g_sdp.value:.6f}")
    
    x_val = x_sdp.value
    X_val = X_sdp.value
    
    evals = np.linalg.eigvalsh(X_val)
    rank_X = np.sum(evals > 1e-4)
    
    gap_matrix = X_val - (x_val @ x_val.T)
    gap_norm = np.linalg.norm(gap_matrix, 'fro')
    
    print(f"Rank of X: {rank_X} (Target: 1)")
    print(f"Relaxation Gap ||X - xx^T||: {gap_norm:.6f}")
    
    if gap_norm > 1e-3:
        print("-> The relaxation is NOT tight (Strict Inequality X > xx^T).")
        print("-> The value obtained is a Lower Bound, not a feasible portfolio.")
    else:
        print("-> The relaxation is tight (X approx xx^T). Global Optimum found!")

    try:
        print(f"Gap vs Local Solver (SLSQP): {final_obj_slsqp - g_sdp.value:.6e}")
    except:
        pass

else:
    print(f"SDP Failed: {prob_sdp.status}")