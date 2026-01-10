import cvxpy as cp
import numpy as np
from control import dlqr

from .MPCControl_base import MPCControl_base

class MPCControl_x(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6, 9])
    u_ids: np.ndarray = np.array([1])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        super()._setup_controller()
        A, B = self.A, self.B
        nx, nu, N = self.nx, self.nu, self.N

        Q = np.diag([10.0, 10.0, 10.0, 50.0])
        R = np.array([[1.0]])
        S = 1e4
        self.Q = Q
        self.R = R

        rho_x = cp.Variable((N+1,), nonneg=True)
        # constraints
        self.constraints += [self.X[:, 0] == self.x0_param]

        # dynamics
        for k in range(N):
            self.constraints += [self.X[:, k + 1] == A @ self.X[:, k] + B @ self.U[:, k]]

        self.constraints += [
            self.X[1, :] <= 0.1745 + rho_x,
            self.X[1, :] >= -0.1745 - rho_x,
            self.U <= 0.26,
            self.U >= -0.26
        ]

        self.objective = 0
        for k in range(self.N):
            self.objective += cp.quad_form(self.X[:, k], Q)
            self.objective += cp.quad_form(self.U[:, k], R)
        self.objective += S * cp.sum_squares(rho_x)

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