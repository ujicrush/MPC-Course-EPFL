import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
import matplotlib.pyplot as plt

from .MPCControl_base import MPCControl_base


class MPCControl_z(MPCControl_base):
    # z-velocity, z-position
    x_ids: np.ndarray = np.array([8, 11])
    u_ids: np.ndarray = np.array([2])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        super()._setup_controller()
        A, B = self.A, self.B
        nx, nu, N = self.nx, self.nu, self.N
        self.Z = cp.Variable((nx, N + 1), name="Z")
        self.V = cp.Variable((nu, N), name="V")

        # cost
        Q = np.diag([100.0, 350.0])
        R = np.array([[1.0]])
        self.Q = Q
        self.R = R

        # LQR feedback for tube
        K, Qf, _ = dlqr(A, B, Q, R)
        self.K = -K
        self.Qf = Qf

        Bcol = B.reshape(nx, 1)
        A_cl = A + Bcol @ self.K
        
        # disturbance set in state space: W_e = B * [-15, 5]
        w_min, w_max = -15.0, 5.0
        v1 = (Bcol * w_min).reshape(-1)
        v2 = (Bcol * w_max).reshape(-1)

        W_e = Polyhedron.from_Vrep(V=np.vstack([v1, v2]))
        W_e.minHrep()
        self.W = W_e

        def mrpi_zonotope_2d(A_cl, Bcol, w_min=-15.0, w_max=5.0, max_iter=80, tol=1e-8):
            """
            Compute E ≈ sum_{i=0}^{∞} A_cl^i * (Bcol*[w_min,w_max]) in 2D
            using 2D zonotope vertex construction (no 2^k blow-up).
            """
            nx = A_cl.shape[0]
            assert nx == 2, "This zonotope shortcut assumes 2D state."

            # segment: c + [-g, g]
            c = (Bcol * ((w_min + w_max) / 2.0)).reshape(-1)           # (2,)
            g = (Bcol * ((w_max - w_min) / 2.0)).reshape(-1)           # (2,)

            # accumulate center and generators
            C = np.zeros(nx)
            G_list = []

            A_pow = np.eye(nx)
            for i in range(max_iter):
                # add A^i * segment
                C = C + A_pow @ c
                Gi = A_pow @ g

                if np.linalg.norm(Gi, 2) > 1e-12:   # ignore near-zero generators
                    G_list.append(Gi)

                # next power
                A_pow = A_pow @ A_cl

                # convergence: when A^i is tiny, further terms negligible
                if np.linalg.norm(A_pow, 2) < tol:
                    break

            if len(G_list) == 0:
                # no disturbance effectively
                return Polyhedron.from_Vrep(V=C.reshape(1, -1))

            G = np.array(G_list)  # shape (m,2)

            # --- 2D zonotope vertices ---
            # angles in [0, pi) (central symmetry)
            ang = np.arctan2(G[:, 1], G[:, 0])
            ang = np.mod(ang, np.pi)
            order = np.argsort(ang)
            Gs = G[order]

            # start at v0 = C - sum Gs
            v0 = C - np.sum(Gs, axis=0)

            verts = [v0.copy()]
            v = v0.copy()
            # forward: add 2*Gi
            for Gi in Gs:
                v = v + 2.0 * Gi
                verts.append(v.copy())
            # backward: add -2*Gi
            for Gi in Gs:
                v = v - 2.0 * Gi
                verts.append(v.copy())

            V = np.vstack(verts)
            E = Polyhedron.from_Vrep(V=V)
            E.minHrep()
            return E
        
        E = mrpi_zonotope_2d(A_cl, Bcol, w_min=-15.0, w_max=5.0, max_iter=80, tol=1e-8)
        self.E = E

        us = self.us[0]
        u_min, u_max = 40 - us, 80 - us

        U_set = Polyhedron.from_Hrep(
            A=np.array([[ 1.0],
                        [-1.0]]),
            b=np.array([u_max, -u_min])
        )
        self.U_set = U_set

        xs_z = float(self.xs[1])
        # Δz >= -xs_z  ->  -Δz <= xs_z
        X_set = Polyhedron.from_Hrep(A=np.array([[0.0, -1.0]]), b=np.array([xs_z]))
        self.X_set = X_set

        X_tilde = X_set.pontryagin_difference(E)
        self.X_tilde = X_tilde

        KE = E.affine_map(self.K)
        U_tilde = U_set.pontryagin_difference(KE)
        self.U_tilde = U_tilde

        def max_invariant_set(A_cl, X: Polyhedron, max_iter = 100) -> Polyhedron:
            O = X
            iter = 1
            converged = False
            while iter < max_iter:
                Oprev = O
                F, f = O.A, O.b
                # Compute the pre-set
                O = Polyhedron.from_Hrep(np.vstack((F, F @ A_cl)), np.vstack((f, f)).reshape((-1,)))
                O.minHrep()
                if O == Oprev:
                    converged = True
                    break
                print('Iteration {0}... not yet converged.'.format(iter))
                iter += 1
            if converged:
                print('Maximum invariant set successfully computed after {0} iterations.'.format(iter))
            return O	

        X_and_KU_tilde = X_tilde.intersect(Polyhedron.from_Hrep(U_tilde.A @ self.K, U_tilde.b))
        Xf_tilde = max_invariant_set(A_cl, X_and_KU_tilde)
        self.Xf_tilde = Xf_tilde
         
        fig, ax = plt.subplots(1, 2, figsize=(10, 4)) # E
        E.plot(ax[0])
        ax[0].set_title("Minimal RPI set $\\mathcal{E}$")
        ax[0].set_xlabel("$\\Delta v_z$")
        ax[0].set_ylabel("$\\Delta z$")
        Xf_tilde.plot(ax[1])
        ax[1].set_title("Terminal set $\\mathcal{X}_f$")
        ax[1].set_xlabel("$\\Delta v_z$")
        ax[1].set_ylabel("$\\Delta z$")
        plt.tight_layout()
        plt.show()

        # tube initial condition: x0 - z0 ∈ E
        self.constraints += [ E.A @ (self.x0_param - self.Z[:, 0]) <= E.b ]

        # nominal dynamics
        for k in range(N):
            self.constraints += [ self.Z[:, k+1] == A @ self.Z[:, k] + B @ self.V[:, k] ]

        # tightened state constraints on z_k (k=0..N-1)
        for k in range(N):
            self.constraints += [ X_tilde.A @ self.Z[:, k] <= X_tilde.b ]

        # tightened input constraints on v_k (k=0..N-1)
        for k in range(N):
            self.constraints += [ U_tilde.A @ self.V[:, k] <= U_tilde.b ]

        # terminal constraint
        self.constraints += [ Xf_tilde.A @ self.Z[:, N] <= Xf_tilde.b ]

        # --- Objective ---
        self.objective = 0
        for k in range(N):
            self.objective += cp.quad_form(self.Z[:, k], Q)
            self.objective += cp.quad_form(self.V[:, k], R)
        self.objective += cp.quad_form(self.Z[:, N], self.Qf)

        self.ocp = cp.Problem(cp.Minimize(self.objective), self.constraints)
        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE

        # 1) extract substate (absolute)
        x0 = np.asarray(x0).reshape(-1)
        if x0.shape[0] != self.nx:
            x_sub = x0[self.x_ids]
        else:
            x_sub = x0

        # 2) convert to delta
        delta_x0 = x_sub - x_target if x_target is not None else x_sub - self.xs
        self.x0_param.value = delta_x0

        # 3) solve nominal tube MPC for (Z,V)
        # self.ocp.solve(verbose=False)

        self.ocp.solve(warm_start=False, verbose=False, **{"NumericFocus": 3, "BarHomogeneous": 1})

        if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            v0 = np.zeros(self.nu)
            z0 = delta_x0.copy()
            z_traj = np.tile(delta_x0.reshape(-1, 1), (1, self.N + 1))
            v_traj = np.zeros((self.nu, self.N))
        else:
            v0 = np.asarray(self.V[:, 0].value).reshape(self.nu,)
            z0 = np.asarray(self.Z[:, 0].value).reshape(self.nx,)
            z_traj = self.Z.value
            v_traj = self.V.value

        # 4) tube feedback
        e0 = delta_x0 - z0
        delta_u0 = v0 + (self.K @ e0).reshape(self.nu,)

        # 5) back to absolute input
        u0 = delta_u0 + u_target if u_target is not None else delta_u0 + self.us

        # optional: return nominal trajectories (shift to absolute for plotting)
        if u_target is not None and z_traj is not None:
            x_traj = z_traj + x_target.reshape(-1, 1)
            u_traj = v_traj + u_target.reshape(-1, 1)
        elif z_traj is not None:
            x_traj = z_traj + self.xs.reshape(-1, 1)
            u_traj = v_traj + self.us.reshape(-1, 1)
        else:
            x_traj = None
            u_traj = None

        # YOUR CODE HERE
        #################################################
        return u0, x_traj, u_traj

    # Not used in tube MPC
    def setup_estimator(self):
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE
        pass
        # YOUR CODE HERE
        ##################################################


    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE
        pass
        # YOUR CODE HERE
        ##################################################
