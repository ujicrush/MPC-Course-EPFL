import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


class MPCControl_roll(MPCControl_base):
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        super()._setup_controller()

        A, B = self.A, self.B
        nx, nu, N = self.nx, self.nu, self.N

        Q = 50 * np.eye(nx)
        R = 0.1 * np.eye(nu)

        self.Q = Q
        self.R = R

        K, Qf, _ = dlqr(A, B, Q, R)
        self.K = -K
        self.Qf = Qf

        M = np.array([[1.0],
                    [-1.0]])
        m = np.array([20.0,
                    20.0])

        F_term = M @ self.K        # shape (2,2)
        f_term = m

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

        lower, upper = print_interval_from_H(Ff, ff, name="roll_sub_state")

        self.constraints +=[
            self.U <= 20,
            self.U >= -20,
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
