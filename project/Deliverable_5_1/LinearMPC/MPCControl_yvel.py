import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
import matplotlib.pyplot as plt

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

        Q = 0.01 * np.eye(nx)
        Q[0,0] *= 5500
        Q[1,1] *= 1.5
        Q[2,2] *= 3.5
        R = 1.0 * np.eye(nu)

        self.Q = Q
        self.R = R

        _, Qf, _ = dlqr(A, B, Q, R)
        self.Qf = Qf

        for k in range(self.N):
            self.constraints += [
                self.X[:, k + 1] == A @ self.X[:, k] + B @ self.U[:, k]
            ]
            
        self.constraints +=[
            self.X[1, :] <= 0.1745,
            self.X[1, :] >= -0.1745,
            self.U <= 0.26,
            self.U >= -0.26
        ]

        self.objective = 0
        for k in range(self.N):
            self.objective += cp.quad_form(self.X[:, k], Q)
            self.objective += cp.quad_form(self.U[:, k], R)
        self.objective += cp.quad_form(self.X[:, N], Qf)

        self.ocp = cp.Problem(cp.Minimize(self.objective), self.constraints)
        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        if x0.shape[0] != self.nx:
            x_sub = x0[self.x_ids]
        else:
            x_sub = x0

        delta_x0 = x_sub - x_target if x_target is not None else x_sub - self.xs
        self.x0_param.value = delta_x0

        self.ocp.solve(warm_start=True)

        if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            delta_u0 = np.zeros(self.nu)
            delta_x_traj = np.tile(delta_x0.reshape(-1, 1), (1, self.N + 1))
            delta_u_traj = np.zeros((self.nu, self.N))
        else:
            delta_u0 = self.U[:, 0].value
            delta_x_traj = self.X.value
            delta_u_traj = self.U.value

        u0 = delta_u0 + u_target if u_target is not None else delta_u0 + self.us
        u_traj = delta_u_traj + u_target.reshape(-1, 1) if u_target is not None else delta_u_traj + self.us.reshape(-1, 1)
        x_traj = delta_x_traj + x_target.reshape(-1, 1) if x_target is not None else delta_x_traj + self.xs.reshape(-1, 1)
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj