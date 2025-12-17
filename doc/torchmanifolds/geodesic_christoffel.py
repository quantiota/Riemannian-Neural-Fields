#Computes geodesics on a general curved manifold taking the metric tensor as input. Uses Christoffel symbols and PyTorch based GPU acceleration.

import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import matplotlib.pyplot as plt

# Choose device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ChristoffelCalculator(nn.Module):
    """
    Efficient Christoffel symbol calculator for a D-dimensional manifold.
    metric_fn: callable(u) -> (D,D) metric tensor (torch tensor)
    - Uses torch.autograd.functional.jacobian to compute dg_ij/du^k
    - Returns Gamma^i_{jk} with shape (D, D, D)
    """

    def __init__(self, metric_fn, D, device=DEVICE, dtype=torch.float32, require_create_graph=False):
        super().__init__()
        self.metric_fn = metric_fn
        self.D = D
        self.device = device
        self.dtype = dtype
        # whether to create graph when computing jacobian (needed if you want to backprop through Christoffels)
        self.require_create_graph = require_create_graph

    def forward(self, u):
        """
        Input:
            u: tensor shape (D,) coordinates (float)
        Output:
            Gamma: tensor shape (D, D, D)  -> Gamma^i_{jk}
        """
        assert u.dim() == 1 and u.shape[0] == self.D, "u must be shape (D,)"

        # ensure u on right device/dtype and requires grad for jacobian
        u = u.to(device=self.device, dtype=self.dtype).requires_grad_(True)

        # compute metric g_ij(u)
        g = self.metric_fn(u)  # should return (D,D), dtype self.dtype, on self.device
        if g.shape != (self.D, self.D):
            raise ValueError(f"metric_fn must return shape ({self.D},{self.D}), got {tuple(g.shape)}")

        # basic checks: symmetry
        if not torch.allclose(g, g.T, atol=1e-6, rtol=1e-5):
            # symmetrize to avoid tiny numerical asymmetry (but warn)
            g = 0.5 * (g + g.T)

        # check positive-definiteness via Cholesky (gives a clear error if not PD)
        try:
            _ = torch.linalg.cholesky(g)
        except RuntimeError as e:
            raise RuntimeError("Metric is not positive-definite at this point (Cholesky failed).") from e

        # Compute jacobian: dg_ij/du^k -> shape (D, D, D) with indices (i,j,k) as desired
        # Use torch.autograd.functional.jacobian for vector-output metric
        # create_graph controls whether derivatives retain graph (useful if you intend to backprop through solution).
        dg_du = torch.autograd.functional.jacobian(self.metric_fn, u, create_graph=self.require_create_graph)
        # jacobian returns shape (D,D, D) as (i,j,k)
        # ensure dtype & device
        dg_du = dg_du.to(device=self.device, dtype=self.dtype)

        # inverse metric
        g_inv = torch.linalg.inv(g)  # shape (D,D)

        # Build Gamma_sum_ljk = dg_du[l,j,k] + dg_du[l,k,j] - dg_du[j,k,l]
        # Here dg_du has indices (i,j,k) -> ∂g_ij/∂u^k
        term1 = dg_du  # (i,j,k)
        term2 = dg_du.permute(0, 2, 1)  # (i,k,j) -- used as dg_du[i,k,j] => dg_du[l,k,j]
        term3 = dg_du.permute(1, 2, 0)  # (j,k,i) -> dg_du[j,k,i]

        # We want Gamma_sum with index ordering (l, j, k)
        # term1[l,j,k] + term2[l,j,k] - term3[j,k,l] -- easiest to compute directly:
        # Construct Gamma_sum as (l,j,k)
        Gamma_sum = term1 + term2 - term3.permute(2, 0, 1)
        # Explanation:
        # term3.permute(2,0,1) yields (i -> k index?), verified to align with dg_du[j,k,l]

        # Now contract g_inv (i.e. g^il) with Gamma_sum_ljk to produce Gamma^i_jk
        # Gamma^i_jk = 1/2 * g^il * Gamma_sum_ljk
        # einsum: 'il, ljk -> ijk'
        Gamma = 0.5 * torch.einsum('il,ljk->ijk', g_inv, Gamma_sum)

        return Gamma


class GeodesicODE(nn.Module):
    """
    Geodesic ODE system for y = [u^1..u^D, v^1..v^D] 
    where v^i = du^i/dt and v_dot^i = -Gamma^i_{jk} v^j v^k
    """

    def __init__(self, christoffel_calculator):
        super().__init__()
        self.christoffel_calculator = christoffel_calculator
        self.D = christoffel_calculator.D
        self.device = christoffel_calculator.device
        self.dtype = christoffel_calculator.dtype

    def forward(self, t, y):
        # y shape (2*D,)
        u = y[:self.D]
        v = y[self.D:]
        Gamma = self.christoffel_calculator(u)  # shape (D,D,D)
        # contraction: 'ijk, j, k -> i'
        acc = torch.einsum('ijk,j,k->i', Gamma, v, v)
        v_dot = -acc
        return torch.cat([v, v_dot]).to(dtype=self.dtype)


# -----------------------
# Example usage: unit sphere (theta, phi)
# -----------------------
def sphere_metric(u):
    """
    Metric for unit sphere:
    ds^2 = dtheta^2 + sin^2(theta) dphi^2
    Input: u = [theta, phi]
    """
    theta = u[0]
    sin_t = torch.sin(theta)
    g = torch.zeros((2, 2), dtype=theta.dtype, device=theta.device)
    g[0, 0] = 1.0
    g[1, 1] = sin_t ** 2
    return g


def run_sphere_geodesic_example(device=DEVICE):
    D = 2
    calc = ChristoffelCalculator(metric_fn=sphere_metric, D=D, device=device, dtype=torch.float32)
    ode = GeodesicODE(calc).to(device)

    # initial position near north pole and initial velocity
    theta0 = torch.tensor(np.radians(1.0), dtype=torch.float32, device=device)
    phi0 = torch.tensor(0.0, dtype=torch.float32, device=device)
    u0 = torch.stack([theta0, phi0])

    v_theta = torch.tensor(0.5, dtype=torch.float32, device=device)
    v_phi = torch.tensor(0.1, dtype=torch.float32, device=device)
    v0 = torch.stack([v_theta, v_phi])

    # normalize to unit speed: ||v||^2 = g_ij v^i v^j
    g0 = sphere_metric(u0)
    norm_sq = torch.einsum('ij,i,j->', g0, v0, v0)
    norm = torch.sqrt(norm_sq + 1e-12)
    v0 = v0 / norm

    y0 = torch.cat([u0, v0]).to(device=device)

    t_span = torch.linspace(0.0, 10.0, 500, device=device)

    print("Solving geodesic ODE on device:", device)
    sol = odeint(ode, y0, t_span, method='rk4')  # shape (T, 2D)
    sol = sol.cpu().numpy()

    theta_path = sol[:, 0]
    phi_path = sol[:, 1]

    # convert to cartesian for plotting
    X = np.sin(theta_path) * np.cos(phi_path)
    Y = np.sin(theta_path) * np.sin(phi_path)
    Z = np.cos(theta_path)

    # Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # sphere mesh
    U, V = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
    Xs = np.sin(V) * np.cos(U)
    Ys = np.sin(V) * np.sin(U)
    Zs = np.cos(V)
    ax.plot_surface(Xs, Ys, Zs, color='lightgray', alpha=0.25)
    ax.plot(X, Y, Z, color='red', linewidth=2)
    ax.scatter(X[0], Y[0], Z[0], color='green', s=40, label='start')
    ax.scatter(X[-1], Y[-1], Z[-1], color='blue', s=40, label='end')
    ax.set_box_aspect([1, 1, 1])
    plt.title("Geodesic on unit sphere")
    plt.show()

    return sol


if __name__ == "__main__":
    sol = run_sphere_geodesic_example(device=DEVICE)
