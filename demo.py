# %%
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.special import logsumexp

def _mc_stable(z, ln_g):
    """Monte Carlo simulation of Föllmer drift/velocity with logsumexp transformation.

    Args:
        z (array-like): of shape (n_mc, dimension), random variable drawn from standard Gaussian
        ln_g (array-like): of shape (n_mc, 1),

            .. math::

                \\text{drift:} \quad & \log (g(x + \sqrt{1 - t} z)),

                \\text{velocity:} \quad & \log (g(tx + \sqrt{1 - t^2} z)),

            where :math:`g` is the scaled Radon-Nikodym derivative of :math:`{\\nu}` w.r.t. :math:`{\\gamma_d}`.

    Returns:
        array-like: MC estimation of Föllmer drift/velocity
    """
    ln_de = logsumexp(ln_g, axis=0)
    z_plus = np.maximum(z, 0)
    z_minus = np.maximum(-z, 0)
    with np.errstate(divide="ignore"):
        ln_nu_plus = logsumexp(np.log(z_plus) + ln_g, axis=0)
        ln_nu_minus = logsumexp(np.log(z_minus) + ln_g, axis=0)
    ln_r_plus = ln_nu_plus - ln_de
    ln_r_minus = ln_nu_minus - ln_de
    r_hat = np.exp(ln_r_plus) - np.exp(ln_r_minus)
    return r_hat

class GaussianMixture1d():
    """1-dimensional Gaussian mixture distribition.

    Attributes:
        dimension (int): dimension of distribution, 1 in this class
        theta (array-like): of shape (k, ), weight of each mode
        mean_array (array-like): of shape (k, ), mean of each mode
        var_array(array-like): of shape (k, ), variance of mode
        n_mode (int): number of modes
        mean (float): mean of distribution, weight sum of mean_array

    Args:
        dimension (int): dimension of distribution
        theta (array-like): of shape (k, ), weight of each mode
        mean_array (array-like): of shape (k, ), mean of each mode
        var_array(array-like): of shape (k, ), variance of mode
    """

    def __init__(self, dimension, weights, mean_array, var_array, *args, **kwargs):
        self.dimension = dimension
        self.weights = np.array(weights)
        self.n_mode = len(self.weights)
        self.mean_array = np.array(mean_array)
        self.var_array = np.array(var_array)
        self.mean = self.mean_array @ self.weights

    def potential(self, x, *args, **kwargs):
        """Potential function :math:`U(x)`.

        Args:
            x (array-like): of shape (n, 1), n observations of 1-d data

        Returns:
            array-like: of shape (n, 1), potential at x
        """
        # (n, k)
        offset = x - self.mean_array
        # (n, k)
        expo = -0.5 * offset**2 / self.var_array
        # (k, )
        weight = self.weights / np.sqrt(self.var_array) / np.sqrt(2 * np.pi)
        # (n, 1)
        return -logsumexp(a=expo, axis=-1, b=weight, keepdims=True)

    def potential_grad(self, x, *args, **kwargs):
        """Gradient of potential function :math:`{\\nabla}U(x)`.

        Args:
            x (array-like): of shape (n, 1), n observations of 1-d data

        Returns:
            array-like: of shape (n, 1), gradient of potential at x
        """
        # (n, k)
        offset = x - self.mean_array
        # (n, k)
        expo = -0.5 * offset**2 / self.var_array
        # (k, )
        weight = self.weights / np.sqrt(self.var_array) / np.sqrt(2 * np.pi)
        # (n, )
        density = np.exp(expo) @ weight
        # (n, k)
        weight = -weight * offset / self.var_array
        # (n, )
        density_grad = np.sum(np.exp(expo) * weight, axis=-1)
        # (n, 1)
        return (-density_grad / density)[..., None]

    def scaled_density(self, x):
        """Scaled density function at x, scaled by 1/np.sqrt(2*np.pi).

        Args:
            x (array-like): of shape (n, 1), n observations of 1-d data

        Returns:
            array-like: of shape (n, 1), scaled density at x
        """
        # (n, k)
        offset = x - self.mean_array
        # (n, k)
        expo = -0.5 * offset**2 / self.var_array
        # (k, )
        weight = self.weights / np.sqrt(self.var_array)
        # (n, 1)
        return np.sum(np.exp(expo) * weight, axis=-1, keepdims=True)

    def scaled_ratio(self, x, preconditioner=None):
        """Scaled Radon-Nikodym derivative function at x, scaled by 1/np.sqrt(2*np.pi)

        Args:
            x (array-like): of shape (n, 1), n observations of 1-d data

        Returns:
            array-like: of shape (n, 1), scaled Radon-Nikodym derivative at x
        """
        if preconditioner:
            p_mu, p_var = preconditioner["offset"], preconditioner["deviation"] ** 2
        else:
            p_mu, p_var = 0.0, 1.0
        # (n, k)
        offset = x - self.mean_array
        # (n, k)
        expo = -0.5 * offset**2 / self.var_array + \
            0.5 * (x - p_mu) ** 2 / p_var
        # (k, )
        weight = self.weights / np.sqrt(self.var_array) * p_var
        return np.sum(np.exp(expo) * weight, axis=-1, keepdims=True)

    def ln_scaled_ratio(self, x, preconditioner=None):
        """Log of scaled Radon-Nikodym derivative function at x.

        Args:
            x (array-like): of shape (n, 1), n observations of 1-d data

        Returns:
            array-like: of shape (n, 1), log of scaled Radon-Nikodym derivative at x
        """
        if preconditioner:
            p_mu, p_var = preconditioner["offset"], preconditioner["deviation"] ** 2
        else:
            p_mu, p_var = 0.0, 1.0
        # (n, k)
        offset = x - self.mean_array
        # (n, k)
        expo = -0.5 * offset**2 / self.var_array + \
            0.5 * (x - p_mu) ** 2 / p_var
        # (k, )
        weight = self.weights / np.sqrt(self.var_array) * p_var
        # (n, 1)
        return logsumexp(a=expo, axis=-1, b=weight, keepdims=True)

    def velocity_closed(self, x, t, preconditioner=None, *args, **kwargs):
        """Closed-form velocity filed for Föllmer flow.

        Args:
            x (array-like): of shape (n, 1), n observations of 1-d data
            t (float): time

        Returns:
            array-like: of shape (n, 1), closed-form Föllmer velocity :math:`V(x, t)`
        """
        if t == 0:
            return np.repeat(np.atleast_2d(self.mean), x.shape[0], axis=0)
        if preconditioner:
            p_mu, p_var = preconditioner["offset"], preconditioner["deviation"] ** 2
        else:
            p_mu, p_var = 0.0, 1.0
        # (k, )
        var = t**2 * self.var_array + (1 - t**2) * p_var
        # (n, k)
        offset = x - t * self.mean_array - (1 - t) * p_mu
        # (n, k)
        p = np.exp(-0.5 * offset**2 / var) / np.sqrt(var)
        # (n, k)
        g = p * self.weights
        # (n, )
        deno = np.sum(g, axis=-1, keepdims=True)
        # (n, )
        nume = np.sum(-offset / var * g, axis=-1, keepdims=True)
        # (n, 1)
        return (x - p_mu + p_var * nume / deno) / t

np.random.seed(42)
dimension = 1
nsample = 10000
target_cfg = {
    "dimension": dimension,
    "weights": [0.5, 0.5],
    "mean_array": [-1, 1],
    "var_array": [0.1, 0.1]
    }
target = GaussianMixture1d(**target_cfg)
velocity = target.velocity_closed
T = 200
h = 1/T
mean = target.mean_array
var = target.var_array
kap = mean.shape[0]

x_ode = np.empty((T, nsample, dimension))
x_ode[0, ...] = np.random.randn(nsample, dimension)
for i in range(T-1):
    t = i*h
    x_ode[i+1, ...] = x_ode[i, ...] + h * velocity(x_ode[i, ...], t)

x_sde = np.empty((T, nsample, dimension))
x_sde[-1, ...] = x_ode[-1, ...]
for i in range(T-1):
    t = i*h
    x_sde[T-i-2, ...] = x_sde[T-i-1, ...] + x_sde[T-i-1, ...]/(t-1) * h + np.random.randn(1, nsample, 1) * math.sqrt(h)

# %%
colors = ["black", "gainsboro", "grey"]
legend_elements = [Line2D([0], [0], color=colors[0], linewidth=2.5, label='ODE'),
                #    Line2D([0], [0], color=colors[1], linewidth=2.5, label='SDE'),
                   Line2D([0], [0], color=colors[2], linewidth=2.5, label='Characteristics'),]

fig, axes = plt.subplots(1, 3, figsize=(8, 3), gridspec_kw={"width_ratios":[1, 8, 1], "wspace": 0.})
ax = axes[0]
ax.axis("off")
ax.invert_xaxis()
sns.kdeplot(y=x_ode.squeeze()[0, :], ax=ax, c="k")
ax.set_ylim([-3, 3])

ax = axes[1]
# ax.axis("off")
t = np.linspace(0., 1., T)
ax.hist2d(np.tile(t, nsample), x_ode.squeeze().T.flatten(), cmap='Greys', bins=[200, 200], range=[[0, 1,], [-3, 3]])

for i in range(8):
    # ax.plot(t, x_ode[:, i, 0], c='snow')
    ax.plot(t, x_ode[:, i, 0], c=colors[0], linewidth=2.5)

# for i in range(8):
#     # ax.plot(t, x_sde[:, i, 0], c='firebrick')
#     ax.plot(t, x_sde[:, i, 0], c=colors[1], linewidth=2.5)
ax.legend(handles=legend_elements)

for i in range(8, 16):
    ind = np.random.randint(0, T-1)
    # ax.plot(t[:ind], x_ode[:ind, i, 0], c='springgreen')
    ax.plot(t[:ind], x_ode[:ind, i, 0], c=colors[2], linewidth=2.5)

ax.set_xticks([0, 1])
ax.set_xticklabels([r"prior", r"target"])
ax.set_yticklabels([])
ax.set_yticks([])

ax = axes[2]
ax.axis("off")
sns.kdeplot(y=x_ode.squeeze()[-1, :], ax=ax, c="k")
ax.set_ylim([-3, 3])

plt.savefig(f"./asset/demo.pdf", bbox_inches="tight")
# %%
