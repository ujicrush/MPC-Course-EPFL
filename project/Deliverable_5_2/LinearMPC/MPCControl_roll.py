import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
import matplotlib.pyplot as plt

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

        # # for deliverable 3.1-3.2
        # Q = 0.01 * np.eye(nx)
        # Q[0,0] *= 5000
        # Q[1,1] *= 5000
        # R = 0.1 * np.eye(nu)

        # for deliverable 3.3
        # Q = 0.01 * np.eye(nx)
        # Q[0,0] *= 5000
        # Q[1,1] *= 10000
        # R = 0.1 * np.eye(nu)   

        # for deliverable 4.1
        Q = 0.01 * np.eye(nx)
        Q[0,0] *= 100
        Q[1,1] *= 10
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

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))

        proj01 = Xf.projection(dims=(0, 1))
        proj01.plot(ax)

        ax.set_title("Terminal invariant set for roll subsystem", fontsize=14)
        ax.set_xlabel("ω_z")
        ax.set_ylabel("γ")

        plt.tight_layout()
        plt.show()

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