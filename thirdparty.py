# %%
import numpy as np
import nnkcde
import flexcode
from flexcode.regression_models import NN
from cdetools.cde_loss import cde_loss
from scipy.integrate import simps

def m1(x):
    y = x[:, 0]**2 + np.exp(x[:, 1] + x[:, 2]/3) + np.sin(x[:, 3] + x[:, 4]) + np.random.randn(x.shape[0])
    return y[:, None]

def m2(x):
    z = np.random.randn(x.shape[0])
    y = x[:, 0]**2 + np.exp(x[:, 1] + x[:, 2]/3) + x[:, 3] - x[:, 4] + (1 + x[:, 1]**2 + x[:, 4]**2) / 2 * z
    return y[:, None]

def m3(x):
    choices = np.random.rand(x.shape[0], 1)
    y = 0.25*np.random.randn(x.shape[0], 1) + x * np.where(choices>0.5, 1, -1)
    return y

def compute_error(cde_test, x_test, y_grid, y_test_observations):
    ntest = x_test.shape[0]
    mean_hat = np.empty(ntest)
    for k in range(ntest):
        mean_hat[k] = simps(cde_test[k, :]*y_grid, x=y_grid)
    mean = np.mean(y_test_observations, axis=1)
    mean_mse = ((mean_hat - mean)**2).mean()

    std_hat = np.empty(ntest)
    for k in range(ntest):
        std_hat[k] = np.sqrt(simps(cde_test[k, :]*(y_grid-mean_hat[k])**2, x=y_grid))
    std = np.std(y_test_observations, axis=1)
    std_mse = ((std_hat - std)**2).mean()

    return mean_mse, std_mse
# %%
X_DIMs = {1: 5, 2: 5, 3: 1}
MODELs = {1: m1, 2: m2, 3: m3}

seed = 42
ntrain = 50000
ntest = 50
n_mc = 1000
n_repeat = 5

for i in range(1, 4):
    x_dim = X_DIMs[i]
    target_fn = MODELs[i]
    rng = np.random.RandomState(seed)
    x_train = rng.randn(ntrain, x_dim)
    x_test = rng.randn(ntest, x_dim)
    y_train = target_fn(x_train)
    y_test = target_fn(x_test)
    y_test_observations = np.empty((ntest, n_mc))
    for k in range(ntest):
        y_test_observations[k, :] = target_fn(np.repeat(x_test[k:k+1, :], n_mc, axis=0)).squeeze()
    print("="*40)
    print(f"Running simulation #{i}")
    n_grid = 1000

    print("-"*20)
    print(f"NNKCDE")
    model = nnkcde.NNKCDE(k=50)
    model.fit(x_train, y_train)
    y_grid = np.linspace(y_train.min(), y_train.max(), n_grid)
    cde_test = model.predict(x_test, y_grid, bandwidth=0.01)

    # grid search best parameters
    bw_search_vec = [1e-2, 5e-2, 0.1]
    k_search_vec = [10, 20, 50, 100]
    results_search = {}
    for bw in bw_search_vec:
        for k in k_search_vec:
            cde_test_temp = model.predict(x_test, y_grid, k=k, bandwidth=bw)
            cde_loss_temp, std_loss_temp = cde_loss(cde_test_temp, y_grid, y_test)
            message_out = r'Bandwith: %.2f, Neighbors: %d, CDE loss: %4.3f \pm %.2f' % (
                bw, k, cde_loss_temp, std_loss_temp)
            print(message_out)
            results_search[(bw, k)] = (cde_loss_temp, std_loss_temp)
    best_combination = sorted(results_search.items(), key=lambda x: x[1][0])[0]
    print('\nBest CDE loss (%4.3f) is achieved using %d Neighbors and KDE bandwidth=%.2f' % (
        best_combination[1][0], best_combination[0][1], best_combination[0][0]))
    best_k, best_bw = best_combination[0][1], best_combination[0][0]

    mean_mse_vec = np.empty(n_repeat)
    std_mse_vec = np.empty(n_repeat)
    for n in range(n_repeat):
        x_test = rng.randn(ntest, x_dim)
        y_test = target_fn(x_test)
        y_test_observations = np.empty((ntest, n_mc))
        for k in range(ntest):
            y_test_observations[k, :] = target_fn(np.repeat(x_test[k:k+1, :], n_mc, axis=0)).squeeze()
        cde_test = model.predict(x_test, y_grid, k=best_k, bandwidth=best_bw)
        mean_mse, std_mse = compute_error(cde_test, x_test, y_grid, y_test_observations)
        mean_mse_vec[n] = mean_mse
        std_mse_vec[n] = std_mse
    print(f"mean: {mean_mse_vec.mean():4.3f} pm {mean_mse_vec.std():4.3f} \
        standard deviation: {std_mse_vec.mean():4.3f} pm {std_mse_vec.std():4.3f}")

    print("-"*20)
    print(f"FlexCode")
    basis_system = "cosine"
    max_basis = 40
    params = {"k": [5, 10, 15, 20]} 
    model = flexcode.FlexCodeModel(NN, max_basis, basis_system, regression_params=params)
    model.fit(x_train, y_train)
    model.tune(x_test, y_test)
    cde_test, y_grid = model.predict(x_test, n_grid=n_grid)
    y_grid = y_grid.squeeze()

    mean_mse_vec = np.empty(n_repeat)
    std_mse_vec = np.empty(n_repeat)
    for n in range(n_repeat):
        x_test = rng.randn(ntest, x_dim)
        y_test = target_fn(x_test)
        y_test_observations = np.empty((ntest, n_mc))
        for k in range(ntest):
            y_test_observations[k, :] = target_fn(np.repeat(x_test[k:k+1, :], n_mc, axis=0)).squeeze()
        cde_test, y_grid = model.predict(x_test, n_grid=n_grid)
        y_grid = y_grid.squeeze()
        mean_mse, std_mse = compute_error(cde_test, x_test, y_grid, y_test_observations)
        mean_mse_vec[n] = mean_mse
        std_mse_vec[n] = std_mse
    print(f"mean: {mean_mse_vec.mean():4.3f} pm {mean_mse_vec.std():4.3f} \
        standard deviation: {std_mse_vec.mean():4.3f} pm {std_mse_vec.std():4.3f}")
