import cvxpy as cp
import numpy as np
from control import dlqr

from .MPCControl_base import MPCControl_base


class MPCControl_yvel(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7])
    u_ids: np.ndarray = np.array([0])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        super()._setup_controller()

        A, B = self.A, self.B
        nx, nu, N = self.nx, self.nu, self.N

        Q = 0.1 * np.eye(nx)
        Q[0,0] *= 10
        Q[1,1] *= 50
        R = 20 * np.eye(nu)

        self.Q = Q
        self.R = R

        K, P, _ = dlqr(A, B, Q, R)
        self.K = -K
        self.P = P

        self.constraints +=[
            self.X[1, :] <= 0.1745,
            self.X[1, :] >= -0.1745,
            self.U <= 0.26,
            self.U >= -0.26
        ]

        self.objective = 0
        for k in range(self.N):
            self.objective += cp.quad_form(self.X[:, k] - self.x_ref_param, Q)
            self.objective += cp.quad_form(self.U[:, k] - self.u_ref_param, R)
        self.objective += cp.quad_form(self.X[:, self.N] - self.x_ref_param, P)

        self.ocp = cp.Problem(cp.Minimize(self.objective), self.constraints)
        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        u0, x_traj, u_traj = super().get_u(x0, x_target, u_target)
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
