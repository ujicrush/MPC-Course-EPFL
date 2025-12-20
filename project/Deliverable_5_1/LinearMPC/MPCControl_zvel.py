import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    # disturbance estimator
    d_estimate: float
    d_gain: float
    x_estimate: np.ndarray

    def _setup_controller(self) -> None:
        #################################################
        super()._setup_controller()

        A, B = self.A, self.B
        nx, nu, N = self.nx, self.nu, self.N

        # # for deliverable 3.1-3.3
        # Q = 100 * np.eye(nx)
        # R = 3.5 * np.eye(nu)

        # for deliverable 4.1
        Q = 100 * np.eye(nx)
        R = 1.5 * np.eye(nu)

        self.Q = Q
        self.R = R

        K, Qf, _ = dlqr(A, B, Q, R)
        self.K = -K
        self.Qf = Qf

        us = self.us[0]
        
        # --- INPUT CONSTRAINTS ONLY (terminal set DROPPED) ---
        self.constraints +=[
            self.U <= 80 - us,
            self.U >= 40 - us
        ]

        # --- SYSTEM DYNAMICS ---
        self.d_param = cp.Parameter()
        for k in range(N):
            self.constraints += [
                self.X[:, k + 1] == A @ self.X[:, k] + B @ (self.U[:, k] + self.d_param)
            ]

        # --- OBJECTIVE ---
        self.objective = 0
        for k in range(N):
            self.objective += cp.quad_form(self.X[:, k], Q)
            self.objective += cp.quad_form(self.U[:, k], R)
        self.objective += cp.quad_form(self.X[:, N], Qf)

        self.ocp = cp.Problem(cp.Minimize(self.objective), self.constraints)

        self.setup_estimator()
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        x_meas = np.asarray(x0).reshape(self.nx,)

        if not hasattr(self, "x_estimate"):
            self.x_estimate = x_meas.copy()

        # estimator update (use previous applied input)
        self.update_estimator(x_meas, self.u_prev)
        self.d_param.value = self.d_estimate

        # run MPC (keep interface as-is)
        u0, x_traj, u_traj = super().get_u(self.x_estimate, x_target, u_target)

        # store applied input for next estimator step
        self.u_prev = np.asarray(u0).reshape(self.nu,)

        return u0, x_traj, u_traj
        #################################################

    def setup_estimator(self):
        ##################################################
        self.d_estimate = 0.0
        self.d_gain = 0.3
        self.u_prev = np.zeros((self.nu,))
        ##################################################

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        ##################################################
        A, B = self.A, self.B

        # predict next state
        x_pred = A @ self.x_estimate + B @ (u_data + self.d_estimate)

        # estimation error
        error = x_data - x_pred

        # disturbance update (integral observer)
        self.d_estimate += self.d_gain * float(error)

        # state correction
        self.x_estimate = x_pred + error
        ##################################################
