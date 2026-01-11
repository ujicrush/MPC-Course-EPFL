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

        self.d_param = cp.Parameter((1,), name="d_est")

        Q = 50 * np.eye(nx)
        R = 0.1 * np.eye(nu)

        self.Q = Q
        self.R = R

        K, Qf, _ = dlqr(A, B, Q, R)
        self.K = -K
        self.Qf = Qf

        # us = self.us[0]

        # # --- INPUT CONSTRAINTS ONLY (terminal set DROPPED) ---
        # self.constraints +=[
        #     self.U <= 80 - us,
        #     self.U >= 40 - us
        # ]

        self.us_param = cp.Parameter((1,), name="u_s_compensate")
        self.constraints += [
            self.U <= 80 - self.us_param,
            self.U >= 40 - self.us_param,
        ]

        # --- SYSTEM DYNAMICS ---
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

        if x0.shape[0] != self.nx:
            x_sub = x0[self.x_ids]
        else:
            x_sub = x0

        x_meas = np.asarray(x_sub).reshape(self.nx,)

        # estimator update (use previous applied input)
        self.update_estimator(x_meas, self.u_prev)
        self.d_param.value = np.array([self.d_estimate], dtype=float)

        if u_target is not None:
            us = float(np.asarray(u_target).reshape(-1,)[0])
        else:
            # self.us might be full vector -> take z channel
            us = float(np.asarray(self.us).reshape(-1,)[self.u_ids[0]])
        self.us_param.value = np.array([us], dtype=float)

        if x_target is None:
            x_ref = 0.0
        else:
            x_ref = float(np.asarray(x_target).reshape(-1,)[0])

        x_used = self.x_estimate if self.x_estimate is not None else x_meas
        self.x0_param.value = (x_used - x_ref).reshape(self.nx,)

        self.ocp.solve()

        if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            delta_u0 = np.zeros(self.nu)
            delta_x_traj = np.tile(self.x0_param.value.reshape(-1, 1), (1, self.N + 1))
            delta_u_traj = np.zeros((self.nu, self.N))
        else:
            delta_u0 = np.asarray(self.U[:, 0].value).reshape(self.nu,)
            delta_x_traj = np.asarray(self.X.value)
            delta_u_traj = np.asarray(self.U.value)
        
        u0 = delta_u0 + us
        u_traj = delta_u_traj + us  # broadcast over horizon
        x_traj = delta_x_traj + x_ref

        self.u_prev = np.asarray(delta_u0).reshape(self.nu,)

        return u0, x_traj, u_traj
        #################################################

    def setup_estimator(self):
        ##################################################
        self.d_estimate = 0.0
        self.d_gain = 3
        self.u_prev = np.zeros((self.nu,))
        self.x_estimate = None
        ##################################################

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        ##################################################
        A, B = self.A, self.B

        if self.x_estimate is None:
            self.x_estimate = x_data.copy()
            return
        
        # predict next state
        x_pred = A @ self.x_estimate + B @ (u_data + self.d_estimate)

        # estimation error
        error = x_data - x_pred

        # disturbance update (integral observer)
        self.d_estimate += self.d_gain * float(error)

        # state correction
        self.x_estimate = x_pred + error
        ##################################################
