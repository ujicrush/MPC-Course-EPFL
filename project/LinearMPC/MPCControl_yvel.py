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

        # for deliverable 3.1-3.2
        Q = 0.01 * np.eye(nx)
        Q[0,0] *= 10
        Q[1,1] *= 80
        Q[2,2] *= 2
        R = 100.0 * np.eye(nu)

        # # for deliverable 3.3
        # Q = 0.01 * np.eye(nx)
        # Q[0,0] *= 200
        # Q[1,1] *= 80
        # Q[2,2] *= 1
        # R = 100 * np.eye(nu)

        # for deliverable 4.1
        # Q = 0.01 * np.eye(nx)
        # Q[0,0] *= 6000
        # Q[1,1] *= 30
        # Q[2,2] *= 1
        # R = 10 * np.eye(nu)

        self.Q = Q
        self.R = R

        K, Qf, _ = dlqr(A, B, Q, R)
        self.K = -K
        self.Qf = Qf

        # x in X = { x | Fx <= f }
        F_x = np.array([
            [0.0, 1.0, 0.0],
            [0.0,-1.0, 0.0],
        ])
        f_x = np.array([0.1745, 0.1745])

        # u in U = { u | Mu <= m }
        M_u = np.array([[1.0],
                        [-1.0]])
        m_u = np.array([0.26, 0.26])

        F_term = np.vstack([F_x, M_u @ self.K])   # (4,3)
        f_term = np.hstack([f_x, m_u])            # (4,)

        Xf = Polyhedron.from_Hrep(F_term, f_term)
        Acl = A + B @ self.K

        while True:
            Xf_prev = Xf
            F_O, f_O = Xf.A, Xf.b
            pre = Polyhedron.from_Hrep(F_O @ Acl, f_O)
            Xf = Xf.intersect(pre)
            Xf.minHrep(True)
            # _ = Xf.Vrep 
            if Xf == Xf_prev:
                break

        Ff, ff = Xf.A, Xf.b

        # Visualization of terminal invariant set for y-subsystem
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Left: projection on (x0, x1)
        proj01 = Xf.projection(dims=(0, 1))
        proj01.plot(axes[0])
        axes[0].set_title("Projection on (ω_x, α)")
        axes[0].set_xlabel("ω_x")
        axes[0].set_ylabel("α")

        # Right: projection on (x1, x2)
        proj12 = Xf.projection(dims=(1, 2))
        proj12.plot(axes[1])
        axes[1].set_title("Projection on (α, v_y)")
        axes[1].set_xlabel("α")
        axes[1].set_ylabel("v_y")

        fig.suptitle("Terminal invariant set for y-subsystem", fontsize=14)
        plt.tight_layout()
        plt.show()

        self.constraints +=[
            self.X[1, :] <= 0.1745,
            self.X[1, :] >= -0.1745,
            self.U <= 0.26,
            self.U >= -0.26,
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
