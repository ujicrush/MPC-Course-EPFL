import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    # only useful for part 5 of the project
    d_estimate: np.ndarray
    d_gain: float

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        super()._setup_controller()

        A, B = self.A, self.B
        nx, nu, N = self.nx, self.nu, self.N

        Q = 100 * np.eye(nx)
        R = 3.5 * np.eye(nu)

        self.Q = Q
        self.R = R

        K, Qf, _ = dlqr(A, B, Q, R)
        self.K = -K
        self.Qf = Qf

        us = self.us[0]
        M = np.array([[1.0],
                      [-1.0]])
        m = np.array([80.0 - us,
                      -40.0 + us])
        
        F_term = M @ self.K        # shape (2, 1)
        f_term = m                 # shape (2,)

        Xf = Polyhedron.from_Hrep(F_term, f_term)
        Acl = A + B @ self.K

        while True:
            Xf_prev = Xf
            F_O, f_O = Xf.A, Xf.b
            pre = Polyhedron.from_Hrep(F_O @ Acl, f_O)
            Xf = Xf.intersect(pre)
            # Xf.minHrep(True)
            # _ = Xf.Vrep 
            if Xf == Xf_prev:
                break

        Ff, ff = Xf.A, Xf.b

        def print_interval_from_H(F, f, name="x"):
            F = np.asarray(F).reshape(-1)
            f = np.asarray(f).reshape(-1)

            lowers = []
            uppers = []

            for a_i, b_i in zip(F, f):
                if np.isclose(a_i, 0.0):
                    if b_i < -1e-9:
                        print("Infeasible constraint: 0 * x <= b but b < 0")
                    continue

                if a_i > 0:
                    # a x <= b  → x <= b/a
                    uppers.append(b_i / a_i)
                else:
                    # a x <= b  → x >= b/a  (a<0)
                    lowers.append(b_i / a_i)

            lower = max(lowers) if lowers else -np.inf
            upper = min(uppers) if uppers else np.inf

            print(f"Terminal invariant set for {name}:")
            print(f"{lower:.4f} <= {name} <= {upper:.4f}")

            return lower, upper

        lower, upper = print_interval_from_H(Ff, ff, name="z_sub_state")

        self.constraints +=[
            self.U <= 80 - us,
            self.U >= 40 - us,
            Ff @ self.X[:, self.N] <= ff
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
        u0, x_traj, u_traj = super().get_u(x0, x_target, u_target)
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj

    def setup_estimator(self):
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE

        self.d_estimate = ...
        self.d_gain = ...

        # YOUR CODE HERE
        ##################################################

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE
        self.d_estimate = ...
        # YOUR CODE HERE
        ##################################################
