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

        Q = 100 * np.eye(nx)
        R = 3.5 * np.eye(nu)

        self.Q = Q
        self.R = R

        # LQR gain (used only for intuition, not terminal set)
        K, _, _ = dlqr(A, B, Q, R)
        self.K = -K

        us = self.us[0]

        # --- INPUT CONSTRAINTS ONLY (terminal set DROPPED) ---
        self.constraints += [
            self.U <= 80 - us,
            self.U >= 40 - us
        ]

        # --- OBJECTIVE ---
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
        # x0 is a MEASUREMENT
        x_meas = x0.copy()

        # initialize estimator on first call
        if not hasattr(self, "x_estimate"):
            self.x_estimate = x_meas.copy()

        # update disturbance estimator
        self.update_estimator(x_meas, self.u_prev)

        # use estimated state for MPC
        u0, x_traj, u_traj = super().get_u(
            self.x_estimate, x_target, u_target
        )

        # store applied input
        self.u_prev = u0.copy()

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
        x_pred = A @ self.x_estimate + B @ u_data + B * self.d_estimate

        # estimation error
        error = x_data - x_pred

        # disturbance update (integral observer)
        self.d_estimate += self.d_gain * error.item()

        # state correction
        self.x_estimate = x_pred + error
        ##################################################
