import math

import torch


class Flow:
    def __init__(self, args):
        self.args = args

    def alpha(self, t):
        raise NotImplementedError

    def beta(self, t):
        raise NotImplementedError

    def alpha2(self, t):
        raise NotImplementedError

    def beta2(self, t):
        raise NotImplementedError

    def dalpha(self, t):
        raise NotImplementedError

    def dbeta(self, t):
        raise NotImplementedError

    @staticmethod
    def prior_sampling(shape, device):
        return torch.randn(shape, device=device)

    def matching_loss_fn(self, velocity_fn, x, y, eps=1e-5):
        """Velocity matching."""
        t = eps + (1-eps) * torch.rand(size=(x.shape[0],), device=x.device)
        at, bt, da, db = self.alpha(t), self.beta(t), self.dalpha(t), self.dbeta(t)
        z = torch.randn_like(x, device=x.device)
        xt = at * z + bt * x
        pred = velocity_fn(xt, t, y)
        target = da * z + db * x
        return (pred - target) ** 2

    @torch.no_grad()
    def solve(self, velocity_fn, x, y):
        solver = self.args.ode_solver
        t_steps = torch.linspace(self.args.eps, self.args.T, self.args.N, device=x.device)
        xt = x
        for i in range(self.args.N-1):
            t = torch.ones(x.shape[0], device=x.device) * t_steps[i]
            dt = t_steps[i+1] - t_steps[i]
            if solver.lower() == "euler":
                xt = self.euler_step(velocity_fn, xt, t, y, dt)
            elif solver.lower() == "heun":
                xt = self.heun_step(velocity_fn, xt, t, y, dt)
            else:
                raise ValueError(f"unrecognized solver {solver}")
        return xt

    @staticmethod
    def euler_step(velocity_fn, x, t, y, dt):
        return x + velocity_fn(x, t, y) * dt

    @staticmethod
    def heun_step(velocity_fn, x, t, y, dt):
        v = velocity_fn(x, t, y)
        t_phi = t + dt
        x_phi = x + v * dt
        v_phi = velocity_fn(x_phi, t_phi, y)
        return x + (v + v_phi) / 2 * dt


class Follmer(Flow):
    def __init__(self, args):
        super().__init__(args)

    def alpha(self, t):
        return ((1 - t ** 2).sqrt())[:, None]

    def beta(self, t):
        return t[:, None]

    def alpha2(self, t):
        return (1 - t ** 2)[:, None]

    def beta2(self, t):
        return (t ** 2)[:, None]

    def dalpha(self, t):
        return (-t / (1 - t ** 2).sqrt())[:, None]

    def dbeta(self, t):
        return (torch.ones_like(t, device=t.device))[:, None]
    
    def matching_loss_fn(self, velocity_fn, x, y, eps=1e-5):
        """Velocity matching."""
        t = eps + (1-eps) * torch.rand(size=(x.shape[0],), device=x.device)
        at, bt = self.alpha(t), self.beta(t)
        z = torch.randn_like(x, device=x.device)
        xt = at * z + bt * x
        pred = (1-t**2).sqrt()[:, None] * velocity_fn(xt, t, y)
        target = -t[:, None] * z + (1-t**2).sqrt()[:, None] * x
        return (pred - target) ** 2


class Linear(Flow):
    def __init__(self, args):
        super().__init__(args)

    def alpha(self, t):
        return (1 - t)[:, None]

    def beta(self, t):
        return t[:, None]

    def alpha2(self, t):
        return ((1 - t) ** 2)[:, None]

    def beta2(self, t):
        return (t ** 2)[:, None]

    def dalpha(self, t):
        return -1. * (torch.ones_like(t, device=t.device))[:, None]

    def dbeta(self, t):
        return (torch.ones_like(t, device=t.device))[:, None]
    

class Trigonometric(Flow):
    def __init__(self, args):
        super().__init__(args)

    def alpha(self, t):
        return torch.cos(torch.pi / 2 * t)[:, None]

    def beta(self, t):
        return torch.sin(torch.pi / 2 * t)[:, None]

    def dalpha(self, t):
        return -torch.pi / 2 * torch.sin(torch.pi / 2 * t)[:, None]

    def dbeta(self, t):
        return torch.pi / 2 * torch.cos(torch.pi / 2 * t)[:, None]


class VESDE:
    def __init__(self, args):
        self.sigma_min = args.sigma_min
        self.sigma_max = args.sigma_max
        self.N = args.N
        self.args = args

    def sigma(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def marginal_std(self, t):
        return self.sigma(t)

    def diffusion(self, t):
        return self.sigma(t) * math.sqrt(2 * math.log(self.sigma_max / self.sigma_min))

    def sde(self, x, t):
        drift = torch.zeros_like(x)
        diffusion = self.diffusion(t)
        return drift, diffusion

    def prior_sampling(self, shape, device):
        return torch.randn(shape, device=device) * self.sigma_max

    def perturb_input(self, x0, t):
        z = torch.randn_like(x0)
        std = self.marginal_std(t)[:, None]
        xt = x0 + std * z
        return xt, z

    def matching_loss_fn(self, eps_fn, x, y, eps=1e-5):
        """Noise-predictor matching."""
        t = torch.rand((x.shape[0], 1), device=x.device) * (1 - eps) + eps
        std = self.marginal_std(t)
        z = torch.randn_like(x, device=x.device)
        xt = x + std * z
        pred = eps_fn(xt, t, y)
        target = z
        return (pred - target) ** 2

    @torch.no_grad()
    def solve(self, eps_fn, x, y):
        score_fn = self.eps_to_score(eps_fn)
        timesteps = torch.linspace(1, 0, self.N, device=x.device)
        dt = - 1 / self.N
        xt = x
        for i in range(self.N - 1):
            t = torch.ones((x.shape[0], 1), device=x.device) * timesteps[i]
            if self.args.sde_solver == "euler-maruyama":
                xt, _ = self.euler_maruyama_step(score_fn, xt, t, y, dt)
            else:
                raise ValueError(f"unrecognized solver {self.args.sde_solver}")
        return xt

    def eps_to_score(self, eps_fn):
        def score_fn(x, t, y):
            return - eps_fn(x, t, y) / self.marginal_std(t)

        return score_fn

    def euler_maruyama_step(self, score_fn, xt, t, y, dt):
        z = torch.randn_like(xt, device=xt.device)
        score = score_fn(xt, t, y)
        drift, diffusion = self.sde(xt, t)
        drift = drift - diffusion ** 2 * score
        x_mean = xt + drift * dt
        xt = x_mean + diffusion * (-dt) ** 0.5 * z
        return xt, x_mean
