import cvxpy as cp
import numpy as np
from control import dlqr

from .MPCControl_base import MPCControl_base


class MPCControl_z(MPCControl_base):
    # z-velocity, z-position
    x_ids: np.ndarray = np.array([8, 11])
    u_ids: np.ndarray = np.array([2])

    def _setup_controller(self) -> None:
        #################################################
        A, B = self.A, self.B
        nx, nu, N = self.nx, self.nu, self.N

        # cost
        Q = np.diag([10.0, 50.0])
        R = np.array([[1.0]])

        # LQR feedback for tube
        K, _, _ = dlqr(A, B, Q, R)
        self.K = -K

        self.X = cp.Variable((nx, N + 1))
        self.U = cp.Variable((nu, N))
    self.x0 = cp.Parameter(nx)
        
        # --- Disturbance bound ---
        w_max = 15.0

        # Conservative tightening (tube radius)
        # ||e|| <= w_max / (1 - ||A+BK||)
        Acl = A + B @ self.K
        gamma = np.linalg.norm(Acl, ord=2)
        tube_bound = w_max / max(1e-3, (1 - gamma))

        self.constraints = []

        # Initial condition
        self.constraints += [
            self.X[:, 0] == self.x0
        ]
        
        self.constraints += [
            # tightened ground constraint: z >= 0
            self.X[1, :] >= tube_bound,

            # input limits (tightened)
            self.U <= 80 - tube_bound,
            self.U >= 40 + tube_bound,
        ]

        # --- Objective ---
        self.objective = 0
        for k in range(N):
            self.objective += cp.quad_form(self.X[:, k], Q)
            self.objective += cp.quad_form(self.U[:, k], R)

        self.ocp = cp.Problem(cp.Minimize(self.objective), self.constraints)
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # delta state
        x_nom0 = x0 - self.xs[self.x_ids]

        # solve nominal MPC
        u_nom, x_nom_traj, u_nom_traj = super().get_u(
            x_nom0, np.zeros_like(x_nom0), np.zeros((1,))
        )

        # tube feedback
        u0 = u_nom + self.K @ x_nom0

        return u0, x_nom_traj, u_nom_traj
        #################################################

    # Not used in tube MPC
    def setup_estimator(self):
        pass

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        pass
