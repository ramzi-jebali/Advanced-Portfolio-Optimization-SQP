import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt

def download_finance_data(n_assets=10):

    # Date range
    start = '2016-01-01'
    end = '2019-12-30'

    # Tickers of assets
    assets = ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'MMC', 'JPM',
              'ZION', 'PSA', 'BAX', 'BMY', 'LUV', 'PCAR', 'TXT', 'TMO',
                'MSFT', 'HPQ', 'SEE', 'VZ', 'CNP', 'NI', 'T', 'BA']
    assets.sort()

    # Downloading data
    if n_assets>23:
        print('\nWarning: max number of assets is limited to 23\n')
        n_assets = 23

    # Training
    training_data = yf.download(assets[:n_assets], start=start, end=end, group_by="ticker", auto_adjust=True)

    # Testing
    testing_data = yf.download(assets[:n_assets], start='2020-01-01', end='2020-12-30', group_by="ticker", auto_adjust=True)

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

def compute_moments(y_data):
    # Defining initial inputs
    mu = y_data.mean().to_numpy().reshape(1, -1)
    sigma = y_data.cov().to_numpy()

    return mu, sigma

def compute_metrics(x, training_df, testing_df):
    # Calculating Annualized Portfolio Stats

    var0 = x * (training_df.cov() @ x)
    var0 = var0.sum().to_frame().T
    std0 = np.sqrt(var0* 252)
    ret0 = training_df.mean().to_frame().T @ x * 252

    var = x * (testing_df.cov() @ x)
    var = var.sum().to_frame().T
    std = np.sqrt(var* 252)
    ret = testing_df.mean().to_frame().T @ x * 252


    stats_training  = pd.concat([ret0, std0, var0], axis=0)
    stats_testing = pd.concat([ret, std, var], axis=0)

    #stats = pd.concat([training_metrics, testing_metrics], axis=1)
    stats_training.index = ['Return', 'Std. Dev.', 'Variance']
    stats_testing.index = ['Return', 'Std. Dev.', 'Variance']

    print('\n\nTraining set: 2016 -- 2019')
    print(stats_training)
    print('\nTesting set: 2020')
    print(stats_testing, "\n")

    return stats_training, stats_testing


def montecarlo_sim(num_assets,n_samples, Y):
    ####################################
    # Montecarlo Simulation
    ####################################

    # Montecarlo simulation of portfolio weights
    rs = np.random.RandomState(seed=123)
    s1 = rs.dirichlet([0.1] * num_assets, n_samples)
    s2 = rs.dirichlet([0.25] * num_assets, n_samples)
    s3 = rs.dirichlet([0.5] * num_assets, n_samples)
    s4 = rs.dirichlet([0.75] * num_assets, n_samples)
    s5 = rs.dirichlet([1.0] * num_assets, n_samples)
    s6 = rs.dirichlet([1.5] * num_assets, n_samples)
    s7 = rs.dirichlet([2.0] * num_assets, n_samples)
    s8 = rs.dirichlet([3.0] * num_assets, n_samples)
    sample = np.concatenate([np.identity(num_assets), s1, s2, s3, s4, s5, s6, s7, s8], axis=0)

    # Calculating mean, standard deviation and square root kurtosis of each portfolio
    m = sample.shape[0]
    M_1 = np.mean(Y.to_numpy(), axis=0).reshape(1, -1)
    M_2 = Y.cov().to_numpy()

    c_mean = 252 * M_1 @ sample.T
    c_var = np.zeros(m)
    #c_kurt = np.zeros(m)

    for i in range(0, m):
        c_var[i] =  (252 * sample[i] @ M_2 @ sample[i].T) ** (0.5)
        #c_kurt[i] = (np.kron(sample[i], sample[i]) @ Sigma_4 @ np.kron(sample[i], sample[i]).T) ** (1 / 4)

    return c_mean, c_var

def scatter_plot_port(c_mean, c_var, ret, std):
    ####################################
    # Plotting Portfolios
    ####################################

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax = np.ravel(ax)

    # Plotting Portfolios in mean-standard deviation plane
    cax0 = ax[0].scatter(c_var, c_mean, c=c_mean / c_var, cmap='Spectral')
    ax[0].scatter(std,
                  ret,
                  marker='*',
                  s=2 ** 8,
                  color='tab:red',
                  label='Computed solution')

    plt.xlabel('Standard deviation [%]', fontsize = 14)
    plt.ylabel('Return [%]', fontsize = 14)
    plt.grid()
    plt.legend()

    plt.show()

    return

def portfolio_composition_plot(Y,x):
    ####################################
    # Plotting Portfolios Composition
    ####################################
    import riskfolio as rp

    # Building the portfolio object
    df = dict()
    for i, elements in enumerate(Y):
        df[elements] = x[i]

    dw = pd.DataFrame(data=df)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax = np.ravel(ax)

    rp.plot_pie(w=dw,
                title='Minimum Variance Portfolio',
                others=0.05,
                nrow=25,
                ax=ax[0])

    fig.tight_layout()
    plt.show()

def compute_coefficients(Y, n):
    ####################################
    # Auxiliary functions
    ####################################

    # Function that calculates D_2
    def duplication_matrix(n):
        out = np.zeros((int(n * (n + 1) / 2), n ** 2))
        for j in range(1, n + 1):
            for i in range(j, n + 1):
                u = np.zeros((int(n * (n + 1) / 2), 1))
                u[round((j - 1) * n + i - ((j - 1) * j) / 2) - 1] = 1.0
                E = np.zeros((n, n))
                E[i - 1, j - 1] = 1.0
                E[j - 1, i - 1] = 1.0
                out += u @ E.reshape(-1, 1).T;
        return out.T

    # Function that calculates L_2
    def duplication_elimination_matrix(n):
        out = np.zeros((int(n * (n + 1) / 2), n ** 2))
        for j in range(n):
            e_j = np.zeros((1, n))
            e_j[0, j] = 1.0
            for i in range(j, n):
                u = np.zeros((int(n * (n + 1) / 2), 1))
                row = round(j * n + i - ((j + 1) * j) / 2)
                u[row] = 1.0
                e_i = np.zeros((1, n))
                e_i[0, i] = 1.0
                out += np.kron(u, np.kron(e_j, e_i))
        return out

    # Function that calculates S_4
    def kurt_matrix(Y):
        P = Y.to_numpy()
        T, n = P.shape
        mu = np.mean(P, axis=0).reshape(1, -1)
        mu = np.repeat(mu, T, axis=0)
        x = P - mu
        ones = np.ones((1, n))
        z = np.kron(ones, x) * np.kron(x, ones)
        S4 = 1 / T * z.T @ z
        return S4

    return duplication_matrix(n), duplication_elimination_matrix(n), kurt_matrix(Y)


def markovitz_portfolio(mu, sigma):
    pass


def markovitz_portfolio_probabilistic(mu, sigma):
    pass


def kurtosis_solve(L2, S2, S4, mu, Y):

    pass


def kurtosis_scp_solve(L2, S2, S4, mu, Y, xk):

    pass

if __name__ == '__main__':
    import scipy.stats as ss
    n_assets = 3
    Ytrain, Ytest = download_finance_data(n_assets=n_assets)
    mu, sigma = compute_moments(Ytrain)

    D2, L2, S4 = compute_coefficients(Ytrain, n_assets)
    S2 = D2.T @ D2 @ L2

