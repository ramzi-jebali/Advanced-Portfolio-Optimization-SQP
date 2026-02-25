# Advanced Portfolio Optimization: From Markowitz to Kurtosis Minimization

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![CVXPY](https://img.shields.io/badge/Optimization-CVXPY-green.svg)](https://www.cvxpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This project implements and compares several portfolio construction strategies, moving from classical Mean-Variance frameworks to high-order moment (Kurtosis) optimization. The study specifically addresses the **"stationarity gap"** observed during the 2020 COVID-19 market crash, demonstrating how non-convex optimization can mitigate tail risks.

## 🧮 Mathematical Framework

### 1. Classical Markowitz (Convex QP)
We minimize portfolio variance $\sigma_p^2$ subject to a minimum return $r_{min}$:
$$\min_{x} x^T \Sigma x \quad \text{s.t.} \quad \mu^T x \geq r_{min}, \quad \sum x_i = 1, \quad x \geq 0$$

### 2. Minimum Kurtosis (Non-Convex)
To protect against "fat-tail" risks, we minimize the fourth-order moment (Kurtosis). This involves solving a non-convex problem where the rank-1 constraint $X = xx^T$ is handled via **Sequential Convex Programming (SCP)** or **Semidefinite Programming (SDP) Relaxation**.

## 🚀 Key Implementations
* **Custom SQP/SCP Solver:** Implemented from scratch a Sequential Quadratic Programming solver to handle non-convex constraints ($||x||_2^2 \geq 0.5$ and Kurtosis terms).
* [cite_start]**Monte Carlo Simulations:** Generated $N=1000$ random portfolios to visualize the **Efficient Frontier**[cite: 369, 387].
* [cite_start]**Backtesting & Regime Shifts:** Analyzed performance across two distinct regimes: Training (2016-2019) and Testing (2020-2021 COVID crash)[cite: 331, 343].

## 📊 Performance Analysis
| Metric | Min-Variance (Benchmark) | Min-Kurtosis (SCP) |
|:---:|:---:|:---:|
| **Realized Return (2020-21)** | ~16% | **~18%** |
| **Volatility (Std Dev)** | 24.5% | **22.9%** |
| **Sharpe Ratio (Test)** | 1.06 | **1.21** |

[cite_start]**Key Finding:** The Minimum Kurtosis portfolio proved more robust during the market crash by favoring defensive assets (PSA, MMC) and explicitly penalizing extreme tail-dependence[cite: 496, 497, 499].

## 🛠 Tech Stack
* **Language:** Python
* [cite_start]**Optimization:** CVXPY (ECOS, OSQP, CLARABEL), SciPy (SLSQP) [cite: 254, 505]
* [cite_start]**Data:** yfinance (Real market data for 23 assets) [cite: 118, 510]
* **Visualization:** Matplotlib
