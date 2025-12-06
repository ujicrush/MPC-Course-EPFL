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
        #################################################
        # YOUR CODE HERE
        nx, nu = self.nx, self.nu
        N = self.N
        A, B = self.A, self.B

        self.X = cp.Variable((nx, N + 1), name="X")
        self.U = cp.Variable((nu, N), name="U")

        self.x0_param = cp.Parameter(nx, name="x0")
        self.x_ref_param = cp.Parameter(nx, name="x_ref")
        self.u_ref_param = cp.Parameter(nu, name="u_ref")

        constraints = []
        constraints.append(self.X[:, 0] == self.x0_param)

        for k in range(N):
            constraints.append(self.X[:, k + 1] == A @ self.X[:, k] + B @ self.U[:, k])

        self.constraints = constraints
        # YOUR CODE HERE
        #################################################

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
        #################################################
        # YOUR CODE HERE
        if x0.shape[0] != self.nx:
            x_sub = x0[self.x_ids]
        else:
            x_sub = x0

        if x_target is None:
            x_ref = np.zeros(self.nx)
        else:
            if x_target.shape[0] != self.nx:
                x_ref = x_target[self.x_ids]
            else:
                x_ref = x_target

        if u_target is None:
            u_ref = np.zeros(self.nu)
        else:
            u_ref = u_target

        self.x0_param.value = x_sub
        self.x_ref_param.value = x_ref
        self.u_ref_param.value = u_ref

        self.ocp.solve(solver=cp.GUROBI, warm_start=True)

        if self.ocp.status != cp.OPTIMAL:
            u0 = np.zeros(self.nu)
            x_traj = np.tile(x_sub.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.zeros((self.nu, self.N))
        else:
            u0 = self.U[:, 0].value
            x_traj = self.X.value
            u_traj = self.U.value

        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
