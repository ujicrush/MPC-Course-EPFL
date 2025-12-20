import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete


class MPCControl_base:
    """Complete states indices"""

    x_ids: np.ndarray
    u_ids: np.ndarray

    """Optimization system"""
    A: np.ndarray
    B: np.ndarray
    xs: np.ndarray
    us: np.ndarray
    nx: int
    nu: int
    Ts: float
    H: float
    N: int

    """Optimization problem"""
    ocp: cp.Problem

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        xs: np.ndarray,
        us: np.ndarray,
        Ts: float,
        H: float,
    ) -> None:
        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)
        self.nx = self.x_ids.shape[0]
        self.nu = self.u_ids.shape[0]

        # System definition
        xids_xi, xids_xj = np.meshgrid(self.x_ids, self.x_ids)
        A_red = A[xids_xi, xids_xj].T
        uids_xi, uids_xj = np.meshgrid(self.x_ids, self.u_ids)
        B_red = B[uids_xi, uids_xj].T

        self.A, self.B = self._discretize(A_red, B_red, Ts)
        self.xs = xs[self.x_ids]
        self.us = us[self.u_ids]

        self._setup_controller()

    def _setup_controller(self) -> None:
        nx, nu, N = self.nx, self.nu, self.N
    
        # Decision variables
        self.x = cp.Variable((nx, N + 1))
        self.u = cp.Variable((nu, N))
    
        # Parameters (updated online)
        self.x0_param = cp.Parameter(nx)
        self.xs_param = cp.Parameter(nx, value=self.xs)
        self.us_param = cp.Parameter(nu, value=self.us)
    
        # Weights
        Q = np.eye(nx)
        R = 0.1 * np.eye(nu)
        P = Q
    
        cost = 0
        constraints = []
    
        # Initial condition
        constraints += [self.x[:, 0] == self.x0_param]
    
        for k in range(N):
            cost += cp.quad_form(self.x[:, k] - self.xs_param, Q)
            cost += cp.quad_form(self.u[:, k] - self.us_param, R)
    
            constraints += [
                self.x[:, k + 1] == self.A @ self.x[:, k] + self.B @ self.u[:, k]
            ]
    
        # Terminal cost
        cost += cp.quad_form(self.x[:, N] - self.xs_param, P)
    
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)


    @staticmethod
    def _discretize(A: np.ndarray, B: np.ndarray, Ts: float):
        nx, nu = B.shape
        C = np.zeros((1, nx))
        D = np.zeros((1, nu))
        A_discrete, B_discrete, _, _, _ = cont2discrete(system=(A, B, C, D), dt=Ts)
        return A_discrete, B_discrete

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
        # Update parameters
        self.x0_param.value = x0
        if x_target is not None:
            self.xs_param.value = x_target
        if u_target is not None:
            self.us_param.value = u_target
    
        # Solve
        self.ocp.solve(solver=cp.OSQP, warm_start=True)
    
        if self.ocp.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError("MPC infeasible")
    
        u0 = self.u[:, 0].value
        x_traj = self.x.value
        u_traj = self.u.value
    
        return u0, x_traj, u_traj

